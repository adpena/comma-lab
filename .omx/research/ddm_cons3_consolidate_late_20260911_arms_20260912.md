# ddm_cons3 — consolidating the arms that closed after cons1 and cons2: eight lane rows, six anti-patterns, three appends, six levers

`[no-triality] [p0-ledger-ok]` · research_only=true · score_claim=false · $0 · no launches, no Modal, no scorer, no pointer or seal write

**Frontier, untouched and not measured here:** `composition S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600] (move 48)`.
This arm measured nothing of the score. Every number below is quoted from the arm that measured it — except the
four I re-derived from source and say so — and the pointer is **unmoved**. Consolidation is means, not the end,
and this memo is not progress toward sub-0.12.

Two LIVE arms were running throughout: `ddm_ren1` and `ddm_dpi1`. Nothing in their trees, their directories or
their memos was written, and the one dpi1 file this arm read is its charter, cited for the anti-pattern below.

Laws routed by this arm: six NEW anti-patterns — `refit_rungs_ranked_by_conditioned_mass_v1`,
`noisy_local_proxy_projects_a_gate_on_a_measured_quantity_v1`, `contract_code_landed_without_its_ledger_row_v1`,
`undived_reserve_constant_transferred_across_volume_regimes_v1`, `display_unit_label_read_as_the_true_unit_v1`,
`warm_start_init_dropped_the_quantization_state_v1` — plus two anchors APPENDED to
`hpr1_counted_section_refit_debt_v1` and `model_section_edit_container_break_fee_v1`, and one falsification
APPENDED to `inherited_timing_leg_reinherited_v1`.

---

## 0. The monitor, before and after

`tools/consolidation_debt.py` (the SessionStart hook's tool), read-only, same command both times:

| signal | before | after | note |
|---|---:|---:|---|
| `pile_files` | 102 | **98** | the −4 is this arm's whole record scope; §5 says why it is that small |
| `pile_lines` | 548 | **458** | −16.4 % |
| `signal_ratio` | 12.0 | **10.6** | it stops firing: 84 memos vs 7 registration commits → 8 |
| `landings` | 1 | 1 | unchanged |
| `stale_commits` | 0 | 0 | already cleared by cons2 |
| `ssd_only_code` | 33 blobs | 33 blobs | the cache is 46 h old and **at least half its named sample is stale** — §6 |

The monitor still reads **CONSOLIDATE-NOW**, and on `pile_files` it is right to. §6 says exactly what is left
and who owns it.

---

## 1. The table — arm → records → lane → equations → DSL → task

The scope was the **complement** of cons1's and cons2's §1 tables. Both were read first; not one row they own
is redone here. cons1 owns obx2, rbf1, gpp1, mxo3, pc3's RD-curve arm, sr5 unit 1, pr18, ntb2 and swp4; cons2
owns the 49 arms of 2026-09-10.

| arm | records committed | lane row (level, verdict) | equations / anti-patterns | DSL lever | task row |
|---|---|---|---|---|---|
| **hpr1** *(the shape-rung arm; cons1 could only route it — it was live)* | the move-48 seal-side **MEASURED decode wall-clock** (`t4_direct`, 1,023.26 s) | **NEW** `ddm_hpr1_hpac_receptive_field_shape_rung_20260911` — L2, CLOSED: rung FALSIFIED at INSTANCE scope (+812 B), its CONTROL became move 47 and its composition move 48 | anchor **APPENDED** to `hpr1_counted_section_refit_debt_v1` (n=1 → **n=2**) | `Hpr1QRecalibrationRung` + `Ntb2HpacFrameQuadRounding` — fired/measured/retired · **`Hpr1ConvADilationRung` — NO events, on purpose** | `ddm_cons3::CLOSE_hpr1_shape_rung` → completed |
| **hpr1 moves 47 + 48** | — | both existing rows, **notes backfilled** (they were EMPTY) — L2, LANDED | (the same refit law) | — | covered by the row above |
| **tmx1** | its charter | **NEW** `ddm_tmx1_refit_tail_mixer_tc1_on_current_field_20260911` — L2, CLOSED NEGATIVE, DO NOT FIRE | **`refit_rungs_ranked_by_conditioned_mass_v1`** (NEW) + the two appended anchors | `Tmx1TailMixerRefit` — fired/measured/retired | `ddm_cons3::CLOSE_tmx1` → completed |
| **pr19** | (memos + JSON rows already tracked) | **NEW** `ddm_pr19_risk_gate_identity_class_and_chain_inheritance_20260911` — L2, LANDED amendment, consequence rows HELD | **`noisy_local_proxy_projects_a_gate_on_a_measured_quantity_v1`** (NEW) + falsification **APPENDED** to `inherited_timing_leg_reinherited_v1` | — | `ddm_cons3::CLOSE_pr19` → completed |
| **pr19's landing** *(measured by hpr1)* | — | (carried on both rows above) | **`contract_code_landed_without_its_ledger_row_v1`** (NEW) | — | covered by `CLOSE_pr19` |
| **pc4** | (charter already tracked) | **NEW** `ddm_pc4_pose_carrier_coefficient_resolve_on_move45_20260911` — L0, **RETIRED BEFORE SPAWN** | none — nothing ran, so there is nothing to register | — | `ddm_cons3::CLOSE_pc4` → **cancelled** |
| **sr5 unit 2** | (memo already tracked) | **NEW** `ddm_sr5_ssd_reserve_derivation_proposal_20260911` — L1, PROPOSAL, nothing changed | **`undived_reserve_constant_transferred_across_volume_regimes_v1`** + **`display_unit_label_read_as_the_true_unit_v1`** (both NEW) | — | `ddm_cons3::CLOSE_sr5_reserve_derivation` → completed |
| **pc3** *(move 45's lever)* | — | existing rows are cons1's; not redone | (cons1's `pc3_rung_price_expires_at_pointer_move_v1`) | `Pc3Cap1PredictorRefit` — fired/measured/retired | — |
| **rbf1** | — | existing row is cons1's; not redone | (cons1's `rbf1_post_render_pose_amplitude_tax_v1`) | `Rbf1PostRenderBoundaryTreatment` — fired/measured/retired | — |
| **MAIN / dpi1** *(dpi1 is LIVE — read-only)* | nothing written into its tree | **no lane row** — dpi1 claims its own | **`warm_start_init_dropped_the_quantization_state_v1`** (NEW; the DEFECT only — the CURE's price is dpi1's and stays PENDING) | — | `ddm_cons3::REGISTER_warm_start_depth_state_anti_pattern` → completed |
| **cons1 · cons2 · cons3** | this memo | **3 NEW** rows — L1, COMPLETE, no score axis | — | — | `ddm_cons3::CLOSE_cons1_and_cons2_self_disposition` → completed |

**Commits:** `35c74eece` (records) · `75668dfae` (equations, anti-patterns, DSL levers, both registries) ·
`271a41cbf` (lane registry, lane audit log, task ledger) · this memo.

---

## 2. What I did NOT register, and why that is the finding

The charter handed me a list of laws. **Not one of them needed a new canonical equation**, and saying why is
more useful than a longer registry.

1. **tmx1's result belongs to hpr1's law, and it makes that law WEAKER.** cons1 registered
   `hpr1_counted_section_refit_debt_v1` at n = 1 and wrote hpr1's own bar into its reactivation field: *"a
   SECOND section's refit"*, naming the exact test — the tc1 tail mixer, *"60 B of state prices 118,511 B"*.
   tmx1 ran that test. It **loses**: +20 B on the live base, +22 B on move 47, against a fire bar of −30.04 B.
   Giving that a second equation id would have hidden it; appending it makes the law **n = 2 with one positive
   and one negative instance**, which authorises no refit rung by itself. That is a downgrade, and it is correct.
2. **tmx1's container fact is the fee law's own EXCLUSION clause, measured.**
   `model_section_edit_container_break_fee_v1` already excludes *"sections that are NOT range-coded under a
   match-finding compressor"*. tmx1 measured it: one member, `p`, `compress_type 0` (STORE), 100 B of ZIP
   overhead, and a rider whose length the schema fixes — so stream bytes reach archive bytes **1:1** on all
   four paired rows (+203→+203, +22→+22, +155→+155, +20→+20). The ±34.8 B container lottery does not govern
   that axis. An exclusion with an anchor is worth more than a sixth law.
3. **pr19 refuses an equation in its own words.** *"Both adjudications are contract text over a custody/spend
   gate, not laws of the S-arithmetic. They produce no equation."* So the risk-gate defect is an anti-pattern
   whose **unwind path carries the envelope arithmetic**, and the chain refusal is appended to the anti-pattern
   ntb2's side already produced. Registering an equation here would import an instrument that does not exist —
   the same call cons2 made on the same object, from the other side.
4. **pc4 produces nothing at all.** It never ran. A retired-before-spawn arm gets a lane row and a cancelled
   task, not a law. Registering a "finding" from an arm that did not exist is the purest form of the fake.

---

## 3. The finding that justifies the unit: a consolidation is an arm, and neither sister dispositioned itself

cons1 dispositioned nine arms. cons2 dispositioned forty-nine. **Neither has a lane row.** A search of the lane
registry for either finds the arms they closed and no trace of the closers — so the next reader cannot see that
the work happened, what it decided, or what it left open, without already knowing the memo's filename. That is
the exact orphan shape both arms were spawned to clear, reproduced at the level above them.

Three rows now exist (cons1, cons2, cons3), each carrying its commits, its registered laws, its own corrections
that stand, and what it left owed. It costs three rows and it stops a rediscovery that would otherwise cost a
re-read of two long memos.

The same shape, smaller: **the two hpr1 pointer-move lanes carried EMPTY notes.** Moves 47 and 48 — the live
pointer — had rows with a level, a name, and nothing about mechanism, receipts or what would reopen them. cons1
could not touch them (hpr1 was live), and the arm that landed them had moved on. They now carry the mechanism,
the leg bytes, the route, and the sub-0.12 arithmetic re-derived at move 48.

---

## 4. Four numbers I re-derived rather than quoted, and one that changed the reading

Because an anchor whose numbers do not re-derive is worse than no anchor:

* **sr5's two reserve constants, read from source.** `src/comma_lab/storage_tiers.py:26` is
  `DEFAULT_RESERVE_FREE_GB = 40.0` with **no comment**; `tools/launch_detached_process.py:396` is
  `BOOT_MIN_FREE_GIB = 40.0` with a four-line derivation naming the 2026-09-04 ENOSPC and the 72 GiB swap peak
  **on the boot volume**. The memo's claim is exactly what the files say. The transfer is real.
* **The unit factor.** 2³⁰/10⁹ = **1.073741824**, and `37,965,692,928 B` is `35.358` GiB against `37.97` GB —
  so a target written off a `df -h` display that labels the GB figure `Gi` names **7.4 % more bytes** than it
  appears to. Both of sr5's tables re-derive.
* **The refit decomposition, again, from the other side.** hpac: `(+351) + (−1,238) = −887 B`. Mixer:
  `0 + (+20) = +20 B` — and the zero is structural, not measured-small: `LaneMixer.__init__` refuses anything
  but `41 + FORMAT.size`, so a refit may change 40 values and **cannot grow**. The two instances differ in the
  one quantity tmx1 names: `12,262 / 40 = 306.6×` of fitted capacity, on an identical 118,511 B stream.
* **The chain arithmetic.** `1,140.81 × 1.0803² = 1,331 s > 1,260 s` — two compounded steps of the only
  observed magnitude already exceed the policy limit, which is why "no per-step bound exists" is the refusal's
  load-bearing half and the 8.03 % is not.

**And one reading that changed:** the charter lists pr19's "code-and-row-together LAW" as if it were pr19's
finding. It is not in pr19's memo. It is **hpr1's**, in the composition-and-inheritance wall memo §4: with
pr19's code landed (`a9fd12704`) and its amendment rows deliberately held, `_pf_blob` compares the LIVE file
against the owner commit, so `PREFIRE_CONTRACT_DRIFT_REFUSED` fired and **no new pre-fire intent could be
emitted at all** — 2 of the amendment's 15 implementation rows drifting, one commit responsible. MAIN's note
said the hold would protect an in-flight intent, and it did; it could not let a new one be created. Both halves
of that choice were defensible, which is why the anti-pattern is about **sequencing** and names no culprit.

---

## 5. The DSL leg, and the lever I deliberately left in the duty queue

Six `Lever` factories land in `tac/witness_dsl/cons3_closed_levers_20260912.py`. As with cons1 and cons2,
`lever_registry.completeness().unmapped` is a **trainer-flag** surface and never listed them: **none of the six
flags exists on either trainer** (MEASURED), and all six act on the coder, the encoder, the carrier or the
renderer — surfaces no trainer argparse owns. Each is marked *do not compile into a launch*.

Five carry a verdict and got `fired` + `measured` + `retired` events. **The sixth, `Hpr1ConvADilationRung`, got
none, on purpose.** It is hpr1's R6 — the cheapest unfired rung on the coder, one receiver edit, no shape bit,
on the 50.5 % of the prior's values that `conv_a` holds — and it has **no verdict**. A lever with no verdict
belongs in `never_fired()` and `duty_to_measure()`, where the controller can rank and nag it; retiring it to
tidy a counter would be the orphan dressed as a closure.

```
known_levers()  226 → 232        never_fired()  212 → 213        duty_to_measure()  212 → 213
```

Six levers added, five orphans removed, **one orphan added on purpose** — and that net `+1` is the honest
number. The implementation note cons2 paid a pass for still holds: events key on the **factory** name (the AST
surface), never on the `Lever`'s `name` string.

---

## 6. What I did not do, and why — read this before trusting the monitor's number

**The residual untracked pile is 98 collapsed entries, and none of it is a record of this scope.** It is the
same residue cons1 and cons2 classified and left: the four `submissions/_staging_move*_pr140_swap/` trees
(deliberately untracked — swp4's rule is re-stage, never repoint), `bundle_verify.git/` trees and `.bundle`
files, serializer patch copies and timestamp directories, `CLAIMS_*` / `CALL_LEDGER_*` snapshots, index dumps,
zero-byte markers, and `ddm_mv1_20260910/staged/`.

**Skipped WITHIN this arm's scope, each with its reason:**

| file | bytes | reason |
|---|---:|---|
| `ddm_ntb2_20260911/equations_recall.json` | 3.7 MB | registry dump, rebuildable by `tools/list_canonical_equations.py --json` (named by cons1) |
| `ddm_rbf1_20260911/equations_recall.json` | 3.7 MB | same class |
| `ddm_ntb2_20260911/recall_0.txt` | 8.4 MB | grep dump |
| `ddm_ntb2_20260911/recall_1.txt` | 90 KB | grep dump |
| `ddm_ntb2_20260911/EQUATIONS_RELEVANT.json` | 168 KB | derived slice of the registry |
| `ddm_ntb2_20260911/equations_stderr.txt` | 0 B | empty marker |
| `.omx/research/ddm_ren1_*` (2 memos) | — | **LIVE ARM.** ren1 owns its own records; this arm never touched them |

Nothing was skipped for tripping gitleaks; the staged-secrets scan reported **0 findings** on every batch. No
ExFAT `._` stub appeared in scope.

**`ssd_only_code`: still 33, still cd3's, and the number is an over-count.** I did not run the sweep: it walks
287,242 files and takes ~745 s on the two SSD tiers that `ddm_ren1` and `ddm_dpi1` are writing to right now.
What I could do cheaply, I did — **I hashed all ten blobs the cache actually names and checked each against
git**:

| state | count | what they are |
|---|---:|---|
| **already in git** | **5** | `ddm_gb2/…/residual_archive.py`, `ddm_sm1/…/residual_archive.py`, `ddm_sm1/…/ddm_sm1_stage.py`, `ddm_bnd3/…/ddm_bnd3_address_term.py`, `ddm_cmp1/…/ddm_cmp1_stage.py` — byte-identical to tracked blobs, so their debt is discharged and the 46-h-old cache still counts them |
| **absent, and expected to be** | **3** | two `compress.py` copies under `ddm_g8r_compress_adversarial_review/{negative_controls,review_final_clean2_controls}` (deliberately mutated review controls) and one `f26_inflate.py` inside an `rp1` candidate runtime tree (a per-candidate receiver copy with tracked siblings) |
| **absent and genuinely owed** | **2** | `ddm_ps1_pr140_update_prep/evidence/stage6_block.py` (19,965 B) and `ddm_ps2_…/evidence/stage7_block.py` (16,219 B) — authored code with **no tracked sibling of any name** |

So on the only sample the cache exposes, **half the named debt is already paid and one fifth is real**. I did
not certify or commit any of them: they belong to arms I did not run, and landing or certifying another arm's
unlanded code from outside is precisely the failure `[[moved_labels_are_not_custody]]` records — I cannot
verify what is partial. cd3 keeps the owner's seat, and now has five fewer blobs to look at and two to start
with.

**Eight dirty `.omx/state` files were MAIN's and sister arms'** — `active_lane_dispatch_claims.md`,
`current_focus.md`, `modal_call_id_ledger.jsonl`, `next_catalog_number.txt`, `operator_p0_ledger.jsonl`,
`probe_outcomes.jsonl`, and (already dirty when I arrived) `lane_registry.json` + `lane_maturity_audit.log`. I
staged only the three files this arm wrote into. **Disclosed, because it is not clean:** the staged
`lane_registry.json` unavoidably carries two rows I did not write and found uncommitted — MAIN's move-48 lane
and **`ddm_ren1`'s L0 claim, ren1 being LIVE**. They are committed **unaltered and uncompleted**; the only way
to exclude them would have been to rewrite a live arm's row, which is worse. For the same reason each arm's
terminal **probe outcome** went into its lane-row notes and its task row rather than into `probe_outcomes.jsonl`
while someone else is mid-rewrite of it.

**`.omx/state/lever_activation_ledger.jsonl` is gitignored** (`.gitignore:363`), so the fifteen activation
events live locally by the repo's own design. The six `Lever` factories that make them queryable **are**
committed.

**Two task rows remain unreadable and are not mine:** `1079_pv1_…_20260816` and `1082_ddm_hm1_…_20260816`, both
non-registration events for unknown task ids, dated 2026-08-16. The ledger is append-only and repairing them
means appending a corrected lifecycle for someone else's task; the dangling-transition check passes without them.

---

## 7. Cross-checks I ran on my own registrations

* **tmx1 ↔ hpr1**: the mixer's leg bytes give `0 + 20 = +20 B` and the archive totals give
  `179,131 − 179,111 = +20 B` independently; hpac's give `(+351) + (−1,238) = −887` against
  `179,359 − 180,246 = −887`. Two sections, two decompositions, four independent arithmetics.
* **the fire bar**: `−2e-5 / 6.658589531221714e-07 = −30.036 B`, which re-derives tmx1's −30.04 B — so the
  +20 B row misses by 50.04 B, and the anchor's residual `50.04 / 30.04 = 1.666` is that miss in units of the
  bar, not a manufactured number.
* **pr19**: `1,232.418725255 × 1.198951 = 1,477.62` re-derives the memo's 1,477.61 stress figure;
  `1,800 − 1,477.61 = 322.39` re-derives its 322 s spare; `1,260 − 1,232.418725255 = 27.581` re-derives the
  27.58 s. The legacy projection's residual `(1,367.77 − 1,232.42)/1,232.42 = 0.1098` is how far the noisy
  proxy sat from the measurement it stood in for.
* **sr5**: `(40 − 24)/24 = 0.667` — the enforced floor is two thirds above the largest derived need, which is
  the anchor's residual; `10.429 × 2 + 0.142 = 21.0` re-derives the proposal.
* **rbf1's exposure**: `2,287,200 / 12,196 = 187.5` re-derives the memo's ratio, and I corrected my own first
  wording — they are token-edge **pixels**, not cells.
* **every** `canonical_producers` and `canonical_consumers` path on all six anti-patterns was checked to
  resolve on disk by the registration tool's `--verify`: **none missing**. One producer path was wrong on the
  first write (`experiments/ddm_tmx1_refit_mixer_price.py` for the real
  `experiments/ddm_tmx1_mixer_price.py`) and was corrected **before** the commit, not after.
* `ruff check --select F` clean on all five Python files; two visible review-tracker passes each;
  `tools/triality_drift_detector.py` rc=0. `lane_maturity validate` → **2,427 lanes clean**.
  `check_canonical_task_status_no_dangling_transitions` → OK.

---

## 8. What this arm did NOT establish

* **`refit_rungs_ranked_by_conditioned_mass_v1` refutes a rule; it does not install one.** tmx1's replacement
  quantity — rank by the fitted capacity a section holds against the noise in the objective that fits it — is
  at **n = 2** (one positive, one negative) and is registered as the anti-pattern's *unwind path*, never as a
  law. tmx1's own sentence: *"That is a proposal at n=2, not a law."* A third section's refit is what would
  change that, and the `semantic` member's 29,862 B is the queued candidate whose staleness **still cannot be
  dated** — the store names the checkpoint, not its training data.
* **The identity-class envelope is not closed.** Its ceiling is the max over **declared** legs, so an
  adversarially-minimal declaration gets 1,140.81 s instead of 1,232.42 s. Closing it needs a registry of every
  retained `t4_direct` leg that the contract does not have. pr19's residual F3 is also open by its own
  accounting: total bytes could fall while a model section grows.
* **The warm-start anti-pattern registers a DEFECT, not a cure.** The dropped bit-depth state is measured; the
  price of restoring it is **`ddm_dpi1`'s and is PENDING**. Nothing here claims a byte. tmx1's low-capacity
  refit is the standing reminder that a restoration can also lose.
* **sr5's reserve derivation changes no number, and its concurrency factor of 2 is a judgement.** Four
  simultaneous cold decodes would need ~42 GiB and would *vindicate* 40. Somebody should measure that
  distribution before any value moves.
* **Nothing here moved the exact score.** The pointer sits where MAIN left it, at move 48.
