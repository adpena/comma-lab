---
title: "eb1 — buried-signal eureka sweep: we built the multiplier for our own binding constraint, proved it exact, then fenced it to one engine"
utc: 2026-07-31
lane_id: lane_ddm_eb1_buried_eureka_sweep_20260731
arm: ddm_eb1 (fresh-eyes buried-signal hunter)
charter_premise: "ddm_gc17 — the binding constraint is READ BANDWIDTH over our own corpus, not new physics"
axis: "[macOS-CPU advisory / static artifact custody]"
research_only: true
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
pointer: 0.1910828242 [contest-CPU] UNMOVED
pointer_moved: false
own_vehicle_composed: v4d 0.9639878 @ 360,238 B (advisory)
scorer_slot_used: false
launches: none
paid_dispatch: none
upstream_mutated: false
---

# eb1 — buried-signal eureka sweep

**Answer first — the single highest-conviction surprise.**

`gc17` measured that the campaign's binding constraint is **one scorer slot**. We already built,
proved, and tested an instrument that multiplies exactly that resource — an **exact monotone
prefix obstruction** that refutes a doomed n600 candidate at **32 of 600 pairs (5.33%)** using
integer rational arithmetic, with a test asserting it consumes **zero scorer calls on resume**.
It is on `main`. And it is deliberately fenced to the g111/g120 parsed-stage engine with explicit
false-authority flags (`public_wire_prune_claim: False`, `family_wide_claim: False`), so the
general byte-close / exact-eval path cannot use it.

The fencing was **correct science** — that arm refused to over-claim. But the consequence is that
the campaign's scarcest resource has an unshipped **≈18.75× multiplier** for above-target
candidates sitting one scoping decision away. That is a buried eureka in the precise sense the
operator asked for: not new physics, existing work whose value was lost at the boundary between
"proved" and "available."

**Second-order corollary, $0, derived from two numbers already in MEMORY.** Because all three score
terms are non-negative, `S ≥ seg_term`. The live own-vehicle v4d row has `seg_term = 0.431179`.
Therefore **v4d is provably outside the sub-0.172 box on the seg term alone**, independent of pose
and rate. The obstruction certificate is the formal statement of that, and applied to the live line
it says every current own-vehicle candidate would be refuted at ~32 pairs. Sobering, and cheap.

---

## RANKED TABLE

Rank = (probability real) × (S-units at stake) × (cheapness to settle). Prices in S against
v4d = 0.9639878 @ 360,238 B; bar = `min(0.15, official 0.172)`.

| # | Found | Primary artifact (verified this session) | Why it was lost (mechanism) | Unlock | ONE next measurement | Falsifier | Label |
|---|---|---|---|---|---|---|---|
| **R1** | **Exact monotone prefix obstruction — a scorer-free early-abort certificate, fenced to one engine** | `src/tac/witness_dsl/g111_parsed_g105_stage_selector_v1.py` (TRACKED, on main) + producer `src/tac/witness_dsl/g120_parsed_stage_production_authority_v2.py:560 exact_prepose_obstruction`. Live receipt: `.omx/research/original_taskspace_inverse_witness_codec_20260725/g121_coldroot_exact_monotone_obstruction_receipt_20260727.json` — 5.7 KB, sha256 `090138f63290a6c998102639ed129acc788d178cb8f5455c84fbed136ac04747` (**UNTRACKED**) | **Deliberate correct scoping mistaken for a general capability.** The engine arm honestly refused to generalize (`test_engine_prefix_obstruction_is_scoped_not_public_prune`, `disposition == BLOCKED_SCOPED`). Nothing then re-asked "should the *general* eval path have this?" The proof lived in an untracked receipt; the code lived under a `g111/g120` name no one greps when thinking "make evals cheaper." | **UNPRICED in S** — it is a throughput multiplier, not a score delta. **MEASURED leverage: refutation at 32/600 pairs = 5.33% ⇒ ≈18.75× more above-target candidates screened per scorer slot.** Directly attacks gc17's named binding constraint. | Replay the existing screen offline against the **live v4d composed row's cached per-pair seg disagreement** and record the pair index at which the exact bound first exceeds 0.172. **$0, no scorer slot** (cached counts only). | If no candidate class in the live queue has `seg_term` alone above the bar, the screen never fires and is worthless. **Note this is a NEGATIVE screen only** — it cannot accelerate a near-target candidate, which is the one you most want to run. | **MEASURED** (receipt, code, tests); **DERIVED** (18.75×) |
| **R2** | **`ddm/sh1_integration_20260727` — 30 files never merged, incl. 3 byte-closed `archive.zip` + a full real-evaluator n600 row + 3 major tools** | Branch head `6a77427ca1`, 10 commits ahead of main, **7,916 insertions**. Receipt `.omx/research/ddm_sh1_local_exact_receipt_20260727.json`. Archive 131,620 B sha256 `5e1441180f83a6d1d12dc72b574d6801f815c555ede3ca2f56ccb228bc45c3b3`; raw decode sha256 `a6cee040…`; `evaluate.py` sha256 `7da71a84…`, 600 samples | **Unmerged branch — the exact mechanism that lost `sc2`** (rescued at `b449aae37f`), recurring 4 days later and still open. Invisible to every main-tree grep. Its own memo says `main_landing_review_required: true`; that review never happened. | MEASURED on that branch: CB1 **MyCar carrier joint ΔS −0.0516456 at +319 B** (control 131,301 B → candidate 131,620 B, same base, joint n600 remeasure); WF7 **−1,776 B lossless**, exact state restore, receiver-closed n600 (rate action −0.00118257). **Ancestor-scoped** (composed row S=23.913488) — per L18 the *mechanisms* transfer, the *numbers* do not. Missing tools on main: `hunt_ddm_pf3b_joint_improving_edge.py` (1,033 L), `measure_ddm_cb1_perclass_carrier_byteclose.py` (1,072 L), `run_ddm_wf7_seven_home_stream_waterfill.py`, `ddm_wf7_seven_home_stream_waterfill.py` | MAIN landing review of the branch; then re-ask whether the **MyCar static-mask carrier** and the **seven-home stream waterfill** apply to the live v4d 6-member container (`tokens.dr7t` is 346,478 of 360,238 B = 96.2%). | If the v4d grammar admits no per-class carrier and no multi-home repack, both legs are ancestor-only and the branch is tools-only value. WF7's own typed blocker `WF7_LEG_NON_E4_STATE_CONTAINER` already says the binding is missing. | **MEASURED** (artifacts on disk) |
| **R3** | **995 untracked-but-committable files in the MAIN worktree — incl. 77 production `src/tac` modules with tests — never `git add`ed** | `git ls-files --others --exclude-standard` = **995**. Breakdown: `.omx/runs/` **734** · `.omx/research/original_taskspace_inverse_witness_codec_20260725/` **133 committable** (470 on disk; the 337-file remainder is correctly ignored by `.gitignore:545 **/checkpoints/` and `:537 *.zip`) · `src/tac/` **77** (`witness_dsl` 59 + `witness_control` 16 + 2 tests), **each module with a matching `tests/test_*.py`** · `tools/tests/` 15 · `tools/` ~7 | **Never staged.** No gate catches this: preflight scans tracked files; `lever_registry` ASTs modules it can find; the triality drift-detector watches commits. Untracked-but-present code is invisible to all three **by construction** — this is orphan **grade 5** (BUILT-ELSEWHERE-UNWIRED-HERE) at maximum scale, and memory already flags grade 5 as DOMINANT. | **UNPRICED.** Prevents recurrence of the campaign's #1 recorded loss mechanism. Includes named-capstone modules: `taskspace_whole_archive_allocator.py`, `taskspace_frontier_closure_audit.py`, `taskspace_r10_n600_maximum_inverse_fitter.py`. | Add a **`git status --porcelain | grep '^??'` count** to the costate SENSE digest / SessionStart hook, with a threshold alarm. **$0.** Then triage-commit the 77 src modules (they have tests). | If every untracked module is a dead duplicate of a tracked one, the volume is noise not signal. Test: for each untracked `src/tac/**.py`, check whether a tracked module exports the same public names. | **MEASURED** (counts, ignore-status) |
| **R4** | **Conditional byte-ceiling table — converts the goal into a per-byte distortion budget** | `.omx/research/original_taskspace_inverse_witness_codec_20260725/g102_final_y1_semantic_base_escape_activation_20260727.json` → `.score_surface` (**UNTRACKED**) | Filed inside a receipt whose top-level status is `BLOCKED_PREREQUISITE_NO_HEAVY_LAUNCH` — a **blocked** verdict buried a **useful** table. Classic mis-filing: the negative headline hid the positive payload. | At the measured `d_pose = 1.0184347e-4`, the exact max `d_seg` for S < 0.172: **100,000 B → 7.35012e-4 · 130,000 B → 5.35254e-4 · 187,563 B → 1.51966e-4** (hard ceiling). A ready-made design target for any rate/distortion trade. | Cross-check the three rows against `tac.contest_score.break_even_d_seg` (already on main). **$0.** If they agree, register as a canonical equation. | Arithmetic disagreement with `break_even_d_seg` ⇒ the table is mis-derived and must not be used as a target. | **MEASURED** anchors; **DERIVED** table |
| **R5** | **Discrete-optimizer bake-off with a positive control and an exact determinism repeat — never committed** | `.omx/tmp/codex_worktrees/einstein_kolmogorov_crux_20260719T212159Z/.omx/research/einstein_kolmogorov_crux_runs_20260719/` (**UNTRACKED**, 12 run dirs × `receipt.json`) | Untracked directory inside a codex worktree whose HEAD *is* on main — so the branch looks fully landed, and `git log` shows nothing missing. The most deceptive of all the shapes found. | **Method ranking** (d_seg, lower better): `zero 0.506155` (positive control — instrument proven live) · `baseline 0.0080696` · `global 0.0078998` · `coordinate 0.0075885` · `coordinate_deep(12) 0.0068896` · `dspsa(8) 0.0064992` · `dspsa_deep(32) 0.0059422` · **`hybrid_dspsa32_coordinate12` 0.0056786 = −29.63% vs baseline**. `dspsa_repeat` reproduces `dspsa` to all 17 digits ⇒ deterministic. `validate_*` runs reproduce their sources exactly. | Identify the search actually used in the live correction/realization step; if it is plain greedy or coordinate descent, run a **hybrid DSPSA+coordinate** arm at matched budget. | **n24 — NOT decision evidence** under the ALLERGIC rule; its own `verdict_scope` says `"n24 SegNet-only PDW1 palette arm"`. If the live step already uses a stronger search, this is redundant. | **MEASURED** at n24; ranking is **INFERRED** for any other vehicle |
| **R6** | **Gate-laundering fixes for #154/#344/#351 — 107 adversarial tests passing, uncommitted for 12 days** | Same worktree: 12 modified tracked files, **2,485 insertions** (1,702 in `src/tac/preflight.py`). Seal state in its own words: `0/3` | Uncommitted working tree; the arm self-invalidated its candidate and never re-serialized. | Closes exactly the class **`wi1` re-found on 07-31** ("6 of 14 gates hollow"). The fix pre-dates the rediscovery by 12 days — a second, independent instance of gc17's central thesis. | Re-run the 107 adversarial tests on current main; if green, serialize as one immutable successor and start a fresh 3-pass seal. | Tests fail against current main ⇒ the fix has bit-rotted and must be re-derived, not merged. | **MEASURED** (diffstat, self-reported test counts); test counts **not re-run by me** |
| **R7** | **CLOSURE of gc17 §6.4 — and it is a NEGATIVE** | `.omx/research/blind_coordinate_401_20260711T221447Z.md`; `src/tac/through_r/blind_coordinate.py`; `.omx/research/ddm_pa2_zero_byte_decode_family_DAG_FEED_20260724.md:116` | gc17 could not adjudicate it within budget and named it explicitly. | **No free bytes.** `#401` is built and wired into `tools/levelset_byte_close_and_eval.py:173` but is **default-OFF** (`:692`) and every receipt shows `"active": false`. It saves **exactly 0 B** on the live line because v4d is a **pure generator** — I read the archive: 6 members, no camera-resolution residual section. Two corrections travel with this: **(a) Q1 ≠ #401's blind set** (Q1 = 294,912 DOF at the *scorer* grid; #401 = 230,904 px/frame at the *camera* grid); **(b) `blind ⊂ ker(A)` — do NOT union them (double-count).** | None needed — closed. Re-opens the moment a camera-resolution residual section is introduced. | If a future grammar counts camera pixels, `#401` becomes a real rate lever immediately. | **MEASURED** |

**Non-additivity clause.** R2 and R4 both touch the rate/carrier pool and **compete, they do not sum**.
R1, R3, R6 are apparatus and compose with everything. R5 is a training/search-time lever.

---

## THE ONE NEXT MEASUREMENT

> **Replay the existing exact monotone prefix obstruction against the live v4d composed row's cached
> per-pair seg disagreement counts, and record the pair index at which the exact rational bound first
> exceeds the bar.**

$0, no scorer slot, minutes. Two decisive outcomes: (a) it fires at a small prefix ⇒ the screen
generalizes beyond the g111 engine and should be offered to the general exact-eval path, converting
gc17's binding constraint into an ~18.75× throughput gain for above-target candidates; (b) it does
not fire ⇒ the certificate is genuinely engine-specific and R1 closes with a mechanism.

---

## HONEST NON-FINDINGS AND UNSEARCHED SCOPE

1. **`/Volumes/APDataStore/pact` — NOT SEARCHED because it is NOT MOUNTED.** `ls` returns
   *"No such file or directory."* The charter's blind spot #1 remains open; it needs the volume
   attached. `/Volumes/VertigoDataTier/pact` is mounted and was reached only through paths cited
   by receipts, not swept.
2. **Did not find**, in the scope searched (`src/`, `tools/`, `experiments/` via grep for
   `seg_score_lower_bound|cumulative_disagreement_pixels|monotone_seg_obstruction`), any consumer of
   the prefix obstruction outside the g111/g120 engine. Four importers exist, all inside that engine.
3. **Did not verify** R6's "107 passed / 346 passed" — carried from the arm's own uncommitted memo.
   **INFERRED, not measured by me.**
4. **Did not sweep** the 28 `.claude/worktrees/agent-*` trees individually. I checked all of them
   programmatically for (a) HEAD not reachable from main and (b) dirty tracked files; only the ones
   in the table flagged. A worktree that is clean *and* whose HEAD is on main can still contain
   untracked files — **I did not check untracked files in the other 27 agent worktrees.**
5. **Did not open** the 470-file taskspace tree exhaustively; it was mined by a delegated sweep for
   measured score rows, evaluator runs, ZIPs, and its own verdict. Non-score content is unsurveyed.
6. **Did not price** R1 in S-units. It is a throughput multiplier; converting it to S requires
   assuming what the freed slot measures, which would be fabrication.
7. **No score claim anywhere.** Nothing here went through `upstream/evaluate.py` in this session.
   MAIN holds the scorer slot; I ran no scorer job, no launch, no dispatch.

---

## THE ONE PARAGRAPH

The operator asked to be surprised positively, and the surprise is not a new lever — it is that the
campaign's single scarcest resource already has a proven multiplier sitting on `main` behind an
honest fence: an exact, integer-rational, zero-scorer-on-resume certificate that kills a doomed
n600 candidate at 32 of 600 pairs, built for one engine and never offered to the general eval path;
around it sit 995 uncommitted-but-committable files in the main worktree including 77 tested production modules,
a 10-commit integration branch carrying three byte-closed archives and a real 600-sample evaluator
row, a validated optimizer bake-off with a live positive control, and a gate-laundering fix that
pre-dated its own rediscovery by twelve days. Every one of these was lost at a *boundary* — proved
but not generalized, built but not staged, merged into a branch but not into `main`, filed under a
blocked headline. **This sweep found no new physics and claims none. It found that gc17 was right,
and that the read problem has a much larger surface than the memos do.**

**Pointer 0.1910828242 `[contest-CPU]` UNMOVED. No row above is a score claim.**

---

**Sisters:** `ddm_gc17_retrieval_is_the_binding_constraint_20260731.md` (the premise; this closes its
§6.4 and §6.6) · `blind_coordinate_401_20260711T221447Z.md` · `ddm_pa2_zero_byte_decode_family_DAG_FEED_20260724.md`
· `ddm_sh1_compose_and_local_exact_findings_20260727.md` · `ddm_wf7_seven_home_stream_waterfill_DAG_FEED_20260725.md`
· `designed_stub_is_orphan_signal_and_a_no_fake_violation_20260731.md` (grade 5) ·
`deferral_scatter_no_consolidated_ledger_defer_at_source_rule_20260729.md` (the `sc2` precedent).
