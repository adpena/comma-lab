# ddm_cons2 — consolidating the 2026-09-10 arms: 146 receipts committed, 49 arms dispositioned, seven laws routed

`[no-triality] [p0-ledger-ok]` · research_only=true · score_claim=false · $0 · no launches, no Modal, no scorer, no pointer or seal write

**Frontier, untouched and not measured here:** `composition S 0.13654774984742127 @ 179,359 B [contest-CUDA T4 n600] (move 47)`.
This arm measured nothing. Every number below is quoted from the arm that measured it, and the pointer is
**unmoved** — consolidation is means, not the end, and this memo is not progress toward sub-0.12.

Laws registered or extended by this arm: `lane_surprise_atlas_oracle_ladder_v1`,
`token_edit_composition_subadditive_on_pair_overlap_v1`,
`fitted_scalar_in_receiver_code_is_counted_content_v1`, `derived_listing_inside_identity_digest_v1`,
`inherited_timing_leg_reinherited_v1`, plus one anchor appended to `rate_directed_predistortion_yield_v1`.

---

**Receipt count, exact:** 146 files across seven commits — 145 arm receipts plus this arm's charter. The
per-family counts in the table below are `git show --name-only` over those seven commits, not estimates.

## 0. The monitor, before and after

`tools/consolidation_debt.py` (the SessionStart hook's tool), read-only, same command both times:

| signal | before | after | note |
|---|---:|---:|---|
| `pile_files` | 242 | **101** | −141; the residue is §6 and none of it is a record |
| `pile_lines` | 437 | **456** | +19 — see below, and it is not a regression |
| `signal_ratio` | 16.7 | **14.3** | 100 memos vs 6 registration commits → 100 vs 7 |
| `stale_commits` | 200 | **cleared** | it no longer fires; eight commits refreshed it |
| `landings` | 1 | 1 | unchanged |
| `ssd_only_code` | 33 blobs | 33 blobs | out of charter scope; owned by cd3 — §6 |

`pile_lines` **rose by 19 and that is the right sign**: it counts uncommitted *lines*, and this unit added
50 lane rows whose notes are long on purpose. Mid-run it read 2,647; the disposition commit took it back to
456. A consolidation that drove this number down by writing terser verdicts would have consolidated nothing.

The monitor still reads **CONSOLIDATE-NOW** and it is right to. §6 says exactly what is left and who owns it.

---

## 1. The table — arm → records → lane → equations → DSL → task

Forty-nine arms, all of which were **missing a lane row entirely**. Grouped by family; every row carries its
verdict *with its scope* and what would reopen it.

| family (arms) | records committed | lane rows | equations / anti-patterns | DSL lever | task rows |
|---|---|---|---|---|---|
| **rlc1–rlc4** (the rule-118 cure chain) | **36** receipts (rlc1 18 · rlc2 10 · rlc4 8 · rlc3 **0** — every rlc3 artifact is a patch copy) — two-stage serializer custody, intended-files manifests, bundle verifications, task events, recall chain | 4 NEW, L1/L0 — rlc1 BLOCKED ON TIMING (60 B, outputs byte-identical to move 40) · rlc2 contract STOP · rlc3 rc-17 custody STOP · rlc4 real 180,178 B candidate REFUSED on manifest-bound identity | **`fitted_scalar_in_receiver_code_is_counted_content_v1`** (the blocker rlc1 cures) + **`derived_listing_inside_identity_digest_v1`** (the refusal rlc4 found) | — | 8 rows: 4 closed with commits `ba0110e15` / `a47543199` / `d598ef3b6` / `5d2632ee4`+`f1b9a0dbb`, 4 noted |
| **rlc5** | **20** receipts — 9 PREFIRE_REFUSAL, the move-43 seal's MEASURED decode wall-clock, FALLBACK_RECOVERY, landing addendum, CPU-axis adjudication | existing rows; the chain's outcome now cited | — (routes to the two above) | — | 3 MAIN rows **closed**: the cure became **pointer move 44** |
| **pr8, pr10–pr17** (second-family review) | (memos already tracked) | 9 NEW, L0 — one P0 blocker, two contract adjudications, five ratifications, one freshness policy | pr8 → the rule-118 anti-pattern; pr14 → the identity-digest anti-pattern | — | — |
| **ffi1–ffi6** (contract implementation) | 0 new (their residue is patch copies) | 6 NEW — two ambiguity STOPs, then four landings (ffi3 froze the contract; ffi4 implemented pr14's scope) | (consumers of the two anti-patterns) | — | — |
| **cust1, cpx1–cpx3, swp2/swp3, pk1** (packet + checker) | **0** — every untracked artifact these seven arms left is a `bundle_verify.git/` tree or a patch copy; their memos were already tracked | 7 NEW — cpx2 85/93→91/93, cpx3 81/93→91/93, three publication REFUSALS retained without waiver | — | — | — |
| **gdc1–gdc4** (generator doors) | (memos already tracked) | 4 NEW, L2 — all four FORMULATION-NO-GO against the **94,010 B** door gdc1 itself derived | — (each is a screen verdict against one door, not a law) | `Gdc1OrderedScanlineProgram`, `Gdc2CategoricalCoolChicDistill`, `Gdc3AnisotropicKeyRowRibbon`, `Gdc4RunNativeEndpointGenerator` — all fired/measured/retired | — |
| **ls1 / ls2** (the tail's last hypothesis) | (memos already tracked) | 2 NEW, L2 — CLOSED at the measured level | **`lane_surprise_atlas_oracle_ladder_v1`** (NEW, two anchors) | `Ls1LaneConditionedSurpriseAtlas` | `ddm_ls1::LS2_BOUND` **closed** |
| **mxo1 / mxo2, ntb1, obx1** | (memos already tracked) | 4 NEW — mxo1 closed by threshold (368 B), mxo2 the grep hazard, ntb1 PARTIAL with live obligations, obx1 design selected / direct falsifier refused | — | `Mxo1FreeDecodeTimeContextMixer` | 3 ntb1 legs **cancelled as SUPERSEDED by ntb2** |
| **tc2–tc4, bnd2/bnd3, eb1/eb2** | **33** receipts (tc2 16 · tc3 9 · tc4 8; bnd/eb left none untracked) — tc2's attempt-1 verification failure, tc3's full final chain, tc4's landing receipt and task completion | 6 existing rows + eb1/eb2 NEW | already registered (`lane_boundary_context_map_bound_v1`, `boundary_segment_recode_price_v1`, `partition_description_rate_distortion_lower_bound_v1`) — **not re-registered** | `Tc2LaneBoundaryContextMap`, `Bnd2BoundarySegmentRecode` | tc3/tc4 exact-eval rows **closed with the retraction**; 3 noted |
| **mv1–mv3, sw1, hb1, cd3, dwc1, sr4** | **31** receipts (mv1 13 · hb1 11 · mv3 4 · mv2 2 · sw1 1; cd3/dwc1/sr4 left none untracked) — the mv landing chain, sw1's SSD audit, hb1's pytest/bundle receipts, dwc1's timing legs | 8 NEW | — | — | dwc1 MAIN_LANDING **closed**; 4 noted |
| **vr5–vr8** (storage custody) | **25** receipts (vr8 10 · vr6 7 · vr5 4 · vr7 4) — four censuses, copy/render recovery ledgers, vr5's four serializer receipts (§5) | 4 NEW — vr7 is the apply that deleted 11 moved payloads; vr8 restored 11/11 (27,468,072,000 B) | — | — | 1 noted |
| **sj1 compose-39** *(09-10 measurement, arm already dispositioned)* | — | — | **`token_edit_composition_subadditive_on_pair_overlap_v1`** (NEW) | — | — |
| **rp1 round 2** *(09-10 measurement, arm already dispositioned)* | — | — | anchor **appended** to `rate_directed_predistortion_yield_v1` | — | — |

**Commits:** `207c389a4` · `3754fabbc` · `39b8b643d` · `fef7bca0b` · `e7742cbe9` · `67ada885e` (receipts 1–6/6)
· `c222f3895` (equations, anti-patterns, DSL levers, both registries) · `454e19137` (lane registry, lane audit
log, task ledger) · `faf9a4f15` (vr5 addendum).

---

## 2. Four places where the charter's framing did not survive the memos

The charter handed me seven laws. Two of them were already registered, and two of the numbers attached to the
rest do not say what the charter says they say. Naming that is more useful than a clean table.

1. **The "1,035 B short" oracle is not the receiver-visible one.** The charter reads *"the lane-surprise atlas
   bound (ls1/ls2: oracle 17,534 B receiver-visible, 1,035 B short)"*. Those are **two different rows of one
   table**. ls1's ladder, Miller–Madow, against the 25,899 B demand:

   | information set | plug-in upper B | MM B | MM shortfall | receiver can compute it? |
   |---|---:|---:|---:|---|
   | `tc1_joint` | 11,568.821 | 8,218.071 | 17,680.929 | yes |
   | `receiver_lane` | 22,222.524 | **17,534.216** | **8,364.784** | yes |
   | `granted_previous_row_lane` | 29,809.287 | 24,863.638 | **1,035.362** | **no** |

   The 17,534.216 B belongs to `receiver_lane`; the 1,035.362 B shortfall belongs to
   `granted_previous_row_lane`, an oracle handed a previous-row Lane distance **the decoder does not have**.
   The receiver-visible rung's own shortfall is **8,364.784 B** (or 3,676.476 B against the optimistic
   uncorrected plug-in, which is where the memo's "short by 3,676.476 B" comes from). Pairing the two makes
   the tail look 8× closer to payable than it is. The equation carries the `receiver_visible` flag per rung so
   the pairing cannot be made again by accident.

2. **sj1's printed decomposition does not close.** The memo reports a miss of **+1.62e-06** and decomposes it
   as +5.086e-06 (six seg repairs lost) and −6.66e-07 (bytes), with a third row whose value is the *word*
   "remainder". The two named terms sum to **+4.42e-06**, so that remainder is **−2.80e-06 — 1.7× the miss
   itself**. Two restatements in the same lineage also fail their own arithmetic: "the seg loss is four times
   the byte effect" is **7.64×**, and the memory index's "six seg repairs lost = +5.09e-6 (four fifths)" is
   not four fifths of a +1.62e-06 miss. None of this weakens the finding — the seg term is still the largest
   contributor and still lives *entirely* on the 13 shared pairs — but nobody should quote the two named terms
   as if they added to the miss. `named_decomposition_remainder_s()` returns the gap and says so.

3. **Two of the seven laws were already registered, and registering them again would have been the harm.**
   The container-break one-sample lottery (mean +0.1 B, **sd 34.8 B**) is already the third anchor
   `fe1_container_break_delta_is_a_one_sample_lottery_20260909` on
   `model_section_edit_container_break_fee_v1`. The first-order-token-price law is already
   `rate_directed_predistortion_yield_v1`. A duplicate law is worse than a missing one: two ids for one
   mechanism means a future arm cites whichever it finds and neither gets the other's anchors. So the 09-10
   legs the yield row *lacked* were **appended to it** instead — sj1's add-leg asymmetry (1.2793× modelled,
   against 0.1445–0.2663 returned on the take-out leg) and rp1's round-2 K curve (neutrality **flat** to
   rank 256 at 4.26 %, while only the prize decays 0.477 → 0.056 bits/test, 8.5×). One law, two directions,
   one id.

4. **ntb2's chain break is a contract fact, not an equation.** The charter lists it among the laws. It
   produces no arithmetic of S, so registering it as a canonical equation would import an instrument that does
   not exist. It is registered as an **anti-pattern** instead, because what recurs is the forbidden *move* —
   inheriting from an already-inherited leg, or patching the contract that blocked you. The refusal's cause is
   measured rather than argued: the same leg with `pointer_archive_sha256` set to move 44 **passes** at
   1232.418725255 s, so the refusal is about the pointer, not the candidate's tree.

---

## 3. The finding that justifies the unit: four arms whose record reads backwards

Each of these is invisible from inside the arm that produced it, and each would send the next reader the
wrong way.

**The rlc1→rlc5 chain looks like four failures and ended at the pointer.** rlc1 BLOCKED on timing, rlc2
STOPPED on a contract gap, rlc3 STOPPED on a Git-object denial, rlc4 was REFUSED on receiver identity. Read
in sequence that is a dead family. It is not: rlc5 carried the same counted-rider rule-118 cure to **pointer
move 44 — S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600], commit `99625f32f`** — and it was the
pre-fire contract's first end-to-end pass. Four STOPs were four *contract* refusals, each of which found a
real defect the contract then fixed. The lane rows now say that on every one of the four.

**cpd1's memo says its journal is empty, and the journal is not.** The memo states *"Current live journal has
0 bytes and no active decisions"*, which is true of the moment it was written and false within hours:
`.omx/state/frontier_disqualifications.jsonl` carries MAIN's row at **2026-09-10T08:07:04.222131Z**
disqualifying archive `299a8201662c…` (lane `ddm_tc3_t4_lane_predictor_tail_20260910`) with
`reason_class rule118_content_in_code`. A reader of the memo alone concludes the retraction never happened.

**mxo2's result memo is filed under mxo2's name and titled mxo3.** cons1 found this from the mxo3 side; the
mxo2 side needed the same row. A search for either arm finds half the story. Both rows now carry the real
path and sha (`33f5828d85d3…`).

**hb1 has receipts and no result memo.** `ddm_hb1_*_20260910.md` matches only a charter, so a name-based
search reads the arm as never having run — while `ddm_hb1_20260910/` holds its bundle verification, pytest
and ruff receipts. The lane row records that shape explicitly.

---

## 4. The DSL leg, and the number that shows it worked

Eight levers, all 2026-09-10, none of which had a `Lever` factory. **Seven of the eight are negative**, which
is exactly the class that gets rediscovered: a negative leaves no artifact anyone trips over.

As with cons1, `lever_registry.completeness().unmapped` is a **trainer-flag** surface and never listed these —
**none of the eight flags exists on either trainer** (MEASURED), and most act on the generator, the coder, the
receiver or the encoder, surfaces no trainer argparse owns. They land as DESIGN-STATE `Lever` factories in
`tac/witness_dsl/cons2_closed_levers_20260910.py`, each marked *do not compile into a launch*, each given
`fired` + `measured` + `retired` events so the duty queue carries the verdict instead of the orphan:

```
known_levers()  218 → 226        never_fired()  220 → 212        duty_to_measure()  220 → 212
```

Eight levers added and eight orphans removed. One implementation note, recorded because it cost a pass:
`known_levers()` keys on the **factory** name (AST-derived), not on the `Lever`'s `name` string. Events
written under the `name` string join nothing and move no counter.

---

## 5. Two things about the enumeration itself

**The scope glob hid four receipts.** This unit enumerated its scope as `.omx/research/ddm_*_20260910*`. vr5's
four serializer receipts live in `ddm_vr5_20260909/` — a directory dated when the arm **started**, while its
apply journals and its work are 09-10. They are committed in the addendum (`faf9a4f15`). The file was not the
problem; the boundary was. An arm's receipts are dated at its start, its work when it ran, and a date-glob
over arm directories will always miss the arms that straddle midnight.

**The serializer timestamp directories are excluded without losing anything checkable.** 118 of the residual
files are `serializer*/<timestamp>-<pid>/` directories holding a multi-MB `.bundle` and `.format-patch` beside
a small `receipts.jsonl`. I verified the claim before making it: each arm's `FINAL_HANDOFF.json` — which *is*
committed — carries the bundle's path, byte count and **sha256** (e.g. rlc5's `f64e43d6061aa916…`, 1,044,916 B),
so the patch copies are re-derivable-or-absent, never unrecorded.

---

## 6. What I did not do, and why — read this before trusting the monitor's number

**The residual untracked pile is 101 collapsed entries / 687 files. None of it is a record.** Classified
exhaustively:

| files | class | disposition |
|---:|---|---|
| 294 | `submissions/_staging_move{40,43,44,47}_pr140_swap/` | **deliberately untracked** — staging trees hold archive copies; swp4's rule is re-stage, never repoint |
| 219 | `bundle_verify.git/` trees and `.bundle` files | excluded class (charter) |
| 118 | serializer patch copies, `landing.patch`, `intended-*.patch`, timestamp dirs | excluded class — and §5 proves nothing checkable is lost |
| 22 | `ddm_mv1_20260910/staged/` | copies of tracked repo files, incl. a 4.2 MB `preflight.py` and a 2.4 MB `review_tracker.duckdb` |
| 14 | zero-byte index markers | nothing to record |
| 10 | logs, recall dumps, index snapshots | excluded class (charter) |
| 3 | `CLAIMS_*` / `CALL_LEDGER_*` snapshots | excluded class (charter) |
| 1 | `ddm_mv1_20260910/wire.py` | **named skip** — an in-place mutator FOR the `staged/` tree; it cannot run without it and its real effect landed through mv2/mv3 |

Skipped **within** the arms, with the reason: `equations_recall.json` ×3 (3.7 MB each — registry dumps
rebuildable by `tools/list_canonical_equations.py --json`), `ddm_rlc2_20260910/INDEX_BEFORE.txt` (6.5 MB git
index dump), `ddm_vr8_20260910/reclaim_row_census.jsonl` (2.5 MB row census),
`ddm_rlc5_20260910/custody/{CLAIMS_BEFORE,CLAIMS_AFTER}.txt` + `CALL_LEDGER_BEFORE.jsonl` (1.5–1.8 MB each),
`custody/INTENDED_LANDING.patch` (8.8 MB), `ddm_ntb2_20260911/EQUATIONS_RELEVANT.json` (168 KB derived slice,
already named by cons1), and every zero-byte marker. Nothing was skipped for tripping gitleaks; the
staged-secrets scan reported **0 findings** on every batch.

**`ssd_only_code`: 33 blobs, still owed, and it has a named owner.** cd3's own memo says it: *"The zero-debt
target is not achieved."* Its serializer returned rc 17 and the proposed owner-process gate is retained for
MAIN. That is cd3's lane row, not this arm's scope.

**Nineteen task rows were left deliberately non-terminal.** They are MAIN's (`MAIN_LANDING`, `MAIN_HARVEST`,
`MAIN_EXACT_EVAL`, `main_disqualify`'s siblings, `strict_flip`) or genuinely open (`bnd3::ITEM_2_REOPENED`).
Each got an `append_note` recording the disposition and the evidence a closer would need. **Forcing a terminal
on someone else's row is not consolidation** — it is the same failure as `moved_labels_are_not_custody`:
writing a label I cannot verify.

**Six dirty `.omx/state` files are MAIN's and sister arms'** — `probe_outcomes.jsonl`, `current_focus.md`,
`operator_p0_ledger.jsonl`, `active_lane_dispatch_claims.md`, `next_catalog_number.txt`,
`modal_call_id_ledger.jsonl`. None carries this arm's id. I staged only the five files this arm wrote. For the
same reason each arm's terminal **probe outcome** went into its lane-row notes and its task row rather than
into `probe_outcomes.jsonl` while someone else is mid-rewrite of it.

**`.omx/state/lever_activation_ledger.jsonl` is gitignored** (`.gitignore:363`), so the 24 activation events
live locally by the repo's own design. The eight `Lever` factories that make them queryable **are** committed.

---

## 7. Cross-checks I ran on my own registrations

Because a registration without an anchor is a fake, and an anchor whose numbers do not re-derive is worse:

* **ls1**: `25,899 − 17,534.216 = 8,364.784` and `25,899 − 22,222.524 = 3,676.476` re-derive both shortfalls
  the memo prints; `180,406 − 154,507 = 25,899` re-derives the demand from the archive and the strict cap,
  **and** `180,466 − 154,507 = 25,959` re-derives gdc1's different figure at move 43 — so the two demands in
  the charter are not a contradiction and must not be harmonized. Both modules assert this in code.
* **ls2**: `385.550480 / 25,899 = 0.014887` re-derives the memo's 1.4887 %. The oracle over-promises the
  charged realisation by `17,534.216 / 385.550480 = 45.5×`.
* **sj1**: `(80 − 74)/80 = 0.075` and `13/282 = 0.046` re-derive the memo's 7.5 % and 4.6 %;
  `0.1376357708 − 0.1376341479 = 1.6229e-06` re-derives the +1.62e-06 miss; `5.086e-06 / 6.66e-07 = 7.636`
  refutes the memo's "four times".
* **rp1 / sj1 asymmetry**: the appended anchor's residual `|1.2793 − 1| = 0.2793` is how far the ADD direction
  sits from the first-order model that priced it, against 0.1445–0.2663 on the take-out leg — one object, two
  directions, and neither is 1.
* **every** `canonical_producers` and `canonical_consumers` path on both new equations and all three
  anti-patterns was checked to resolve on disk: one was wrong
  (`ddm_pr18_manifest_row_exclusion_from_behavior_digest_…` vs the real
  `…from_receiver_behavior_digest_…`) and was corrected before the commit, **not** after.
* `ruff check --select F` clean on all five Python files; two visible review-tracker passes each.
* `lane_maturity validate` → **2,417 lanes clean**. `check_canonical_task_status_no_dangling_transitions` → OK.

---

## 8. What this arm did NOT establish

* **`lane_surprise_atlas_oracle_ladder_v1` closes one OBJECT, not a family.** It says the receiver-visible tail
  on the *shipped move-44 field* is closed at the measured level. It does not say a tail model cannot work on
  a different field, and ls2's refusal was **timing**, not gain — a native receiver with a measured decode
  wall-clock inside the 1260 s gate reopens it, and the gate has only 27.581274745 s of margin left.
* **`token_edit_composition_subadditive_on_pair_overlap_v1` is n = 1.** One composition, one overlap fraction.
  The sign and the locus transfer; 6-of-80 and 7.5 % do not. It also does not cover cell-*colliding* edits,
  which are a different and worse case.
* **The 09-10 negatives are formulation-scoped, and I registered no equation for any of them.** gdc1–gdc4 each
  missed one door by a measured margin; that is a screen verdict about a construction, not a law about the
  problem. Reading four NO-GOs as "generators cannot pay" would be exactly the over-generalization the
  verdict-scope ladder exists to prevent.
* **Nothing here moved the exact score.** The pointer sits where MAIN left it, at move 47.
