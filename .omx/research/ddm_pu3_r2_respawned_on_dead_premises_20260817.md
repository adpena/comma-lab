# ddm_pu3 r2 — the pose solve was uncapped 16 days ago, run, dispatched, and REFUSED. I was respawned on the same four dead premises one day after the cure landed.

- arm: `ddm_pu3_uncapped_pose_solve` (**second spawn**, 2026-08-17) · cost **$0** · no dispatch,
  no governed launch, no scorer job. Scorer slot untouched (live MPS training pid 1746).
- axis: `[contest-CUDA T4 n600]` for every frontier component; arithmetic is exact float; premise
  verification is **source**. `score_claim=false`, `promotable=false`. **Pointer UNMOVED.**
- payload: `/Volumes/APDataStore/pact/ddm_pu3/retained/` — `pu3_r2_verification_and_arithmetic.json`
  **8,330 B** sha256 `016e185e44feefbd7424eabd589ccf7dab924e51b8836c7d6cc8700e8bf7cc22` ·
  `pu3_r2_charter_text_recurrence_evidence.md` **603 B** sha256 `97711f0b…` ·
  `pu3_r2_registry_row.jsonl` **3,091 B** sha256 `bc9c072d…`.

## ANSWER FIRST

1. **All four of my charter's premises are false, and I verified every one at source myself
   rather than inheriting the prior arm's read.** The cap was deleted 2026-08-01 (`f13ffdf4b3`);
   **two** convergence tests sit in the same file; the "13–23%/iteration" figure is one n=1
   `STALE_REHEARSAL` receipt falsified four times at larger n; and the uncapped solve was built,
   converged, dispatched to T4 n600 and **REFUSED at +1.686e-02 S**, having made `d_pose`
   **8.93× WORSE**.
2. **The ceiling, stated as the charter requires.** Driving pose to **exactly zero** buys
   **−0.0082945765 S**. The gap to 0.15 is **−0.0095972930**. Pose is **86.43%** of the gap and
   **cannot close it alone** — zeroing it entirely still leaves **0.0013027165** on the table.
   No mechanism has reached any of that 86.43%, and the one that shipped moved pose the wrong way.
3. **The genuinely new fact is that I exist.** The falsified-premise cure landed 2026-08-16 and
   named `charter:ddm_pu3_uncapped_pose_solve` in its own `propagated_into` list. I was respawned
   **one day later with all four premises intact.** I measured why: **two independent causes**, one
   of which nobody had found (coverage), one of which MAIN registered hours before me (reachability).
4. **I closed the coverage half.** The 08-16 row matched only the *percentage*. The sibling CAP and
   NO-CONVERGENCE-TEST claims — in the *same sentence of the same charter* — had no row, so a
   charter could restate three dead premises and draw one warning. New row lands; my charter now
   draws **2 warnings instead of 1**, and a control charter draws **0**.
5. **I executed the free fire-order that pv1 and pu3 both raised and neither ran** — the mis-scoped
   `d_pose` pin. `6.885642960696714e-06` is **CP135 @ 186,252 B**, not hv1; it was live in
   `HV1_D_POSE`, mis-citing `ddm_wc2` as authority when **wc2 says the opposite at its own
   :747-748**. Worth **3.400899e-06 S** of bar error. Fixed by **derivation**, not by copying.

**STORES CONSULTED:** `ddm_pu3_falsified_premise_propagation_20260816.md` ·
`ddm_ps1u_r2_dual_axis_pose_verdict_20260816.md` · `ddm_ps1u_uncapped_pose_solve_20260816.md` ·
`ddm_pv1_pose_floor_and_admission_bar_20260816.md` · `ddm_pg1_pose_gn_convergence_20260802.md` ·
`ddm_ss1_selection_vs_search_20260803.md` · `ddm_wc2_hpac_mps_port_20260814.md` ·
`ddm_eg1_pose_gn_rehearsal_20260728.json` · `ddm_pk3`/`pk4` · `ddm_qs5` · `ddm_ra3` · `ddm_fb1` ·
`.omx/research/falsified_premise_registry.jsonl` (13 rows) ·
`src/tac/optimization/terminal_pose_gn.py` · commit `f13ffdf4b3`.

---

## 1. Premise verification — re-derived at source, not confirmed from the sibling memo

| charter claim | status | source evidence I read myself |
|---|---|---|
| "HARD-CAPPED at 2–3 relinearizations" | **FALSE** | `terminal_pose_gn.py:535` = `_integer(self.relinearizations, "config.relinearizations", minimum=2)` — **no maximum argument exists**. Ceiling deleted by `f13ffdf4b3`, 2026-08-01. |
| "with NO convergence test" | **FALSE — there are two** | `:1196-1216` stop-on-rejection, and `:1217-1244` marginal-value floor. |
| "STILL DESCENDING 13–23% per iteration" | **FALSE — n=1, falsified 4×** | §2 |
| "a MEASURED truncation **nobody uncapped**" | **FALSE — uncapped, run, dispatched, refused** | §3 |

**The convergence test is a proof, not a threshold** — worth quoting, because the charter asserts
its absence and the code argues its own termination:

> STOP ON REJECTION — a proof, not a heuristic threshold. A rejected iteration leaves
> `current_codes` and `current_evaluation` untouched. Every input to the next iteration is
> therefore bit-identical … The next iteration cannot differ, so it cannot succeed.

And the removed cap was replaced by a *derived* bound, not a bigger number — `__post_init__`
records that strict decrease on a finite integer lattice bounds the accepted-step count with no cap
at all. So "uncap it with a derived criterion" — my charter's task 2 — describes work that was
already done, in the form the charter asks for.

**One thing that looks live and is not.** `tools/pb1_terminal_pose_gn_600.py:111` still defaults
`--relinearizations 2` (the floor) after the ceiling was removed. `ddm_pw1` re-derived at source
that `pb1` is unreferenced by any v4d script and `ddm_sv1` confirms the whole `terminal_pose_gn`
chain is off the live path. **Dead code — reporting it as a finding would be a fake.**

## 2. The 13–23% figure: one pair, four falsifications

Origin `.omx/research/ddm_eg1_pose_gn_rehearsal_20260728.json` — `max_pairs: 1`,
`authority_mode: STALE_REHEARSAL`, `score_claim: false`, `production_accepted: false`, and its own
`authority_blocker`: *"Pinned frozen PoseNet was evaluated on exactly one stale composed pair."*
The 13.2164% / 23.2403% are two **final-step** drops.

| receipt | measured | scale |
|---|---|---|
| `ddm_pg1` | 1.2% / relinearization at the shipped bound | n>1 |
| `ddm_ss1` | 1.2% / relinearization (confirms pg1) | n>1 |
| `ddm_pv1` | **0.1549%** forfeited; **0/50 pairs stop on any cap** | n=50 seeded-random |
| `ddm_ps1u` | 0.07% | vehicle |

A 1-pair sample overstated the rate by **~85×–150×** against n=50. This is the n=1/prefix
population law (m88/m96) crossed with cross-regime constant transfer — and note it entered *code*
as motivation (`terminal_pose_gn.py:520`), which is how an advisory n=1 number acquires the look of
a derived constant.

**I did not run a fifth falsification.** pv1 measured **0/50 pairs stopping on any cap** — the
answer is already zero, and a 5th run has near-zero information gain. On the n≥120 bar my charter
sets: that bar exists to defeat **prefix** bias (pose prefixes measure 2.5–4.2× harder). pv1's n=50
was **seeded-random**, so it is unbiased — only lower-powered. Spending a slot to re-confirm zero
would be the duplication the charter warned about, pointed the wrong way.

## 3. The uncapped solve was already dispatched — and refused

`ddm_ps1u` r2, candidate sha `97048f9f…`, 183,347 B on the hv1 base, call
`fc-01M05JNY5VWA152YF1MBKS37HE`, `[contest-CUDA T4 n600]`:

| axis | measurement | ΔS |
|---|---|---:|
| seg | −37 flips (34,970 → 34,933) | −3.136529e-05 |
| rate | +588 B | +3.915251e-04 |
| **pose** | d_pose 6.145931e-05 vs 6.88e-06 = **8.93×** | **+1.649641e-02** |
| **net** | | **+1.685657e-02 → REFUSE** |

Repeat pass identical, so the pose figure is signal. The mechanism was an **assumed zero**:
`POSE_SCREEN_RESULT.json` carried `local_pose_delta: 0.0, pose_unmeasured: true`. The +588 B bought
neither rate nor pose — it cost both.

**So my charter's tasks 2–4 were complete before my first spawn, and refused before my second.**

## 4. The arithmetic, independently re-derived (and the ceiling)

Base **hv1 ep0634**, sha `80d9c8c6…`, 182,759 B, S **0.15959729295498598**.

| term | S | share |
|---|---:|---:|
| rate | 0.121691700 | 76.25% |
| seg | 0.029611000 | 18.55% |
| **pose** | **0.008294577** | **5.20%** |
| **gap to 0.15** | **−0.009597293** | |

- **POSE CEILING = −0.0082945765 = 86.43% of the gap.** Zeroing pose leaves **0.0013027165**.
- **d_pose = 0.0082945765² / 10 = 6.879999931e-06**, carried on the receipt as **6.88e-06**.
- **Break-even screen** for a candidate costing ΔB bytes:
  `f = 1 − (1 − ΔB·6.658589531e-07 / 0.008294576541)²`
  26 B → 0.417% · 31 B → **0.4971%** · 100 B → 1.599% · 588 B → **9.2177%** · 997 B → 15.367% ·
  1,749 B → 26.109%.
  Two independent cross-checks land exactly: **588 B reproduces pv1's 9.2177%**, and **31 B
  reproduces pk3's separately-derived 0.497%**.

## 5. The mis-scoped constant — fire-order raised twice, executed here

`ddm_sr1_manufactured_seg_recovery.py:1442` read:

```
HV1_D_POSE = 6.885642960696714e-06   # ddm_wc2_hpac_mps_port_20260814.md, hv1 ep0634
```

Both halves wrong. **wc2 itself says so at :747-748** — that value is the **CP135 base at
186,252 B**. Proof it cannot be hv1's unrounded source: the 3-significant-figure interval that
rounds to 6.88e-06 is `[6.875e-06, 6.885e-06)`, and `6.8856e-06` falls **outside** it. Carrying it
onto hv1 overstates the pose term by **3.400899e-06 S**.

It was **live** — 8 sites, including `A1_FOA_LIVE_BELOW`. Landed:

- **Fixed by derivation**, not by copying a different receipt: `HV1_D_POSE = 6.88e-06`, with the
  derivation from hv1's own pose term written into the comment, plus the rounding-interval proof.
- **The pre-registered bar `A1_FOA_LIVE_BELOW = 0.0026240` was deliberately NOT moved.**
  `sqrt(6.88e-06) = 0.0026230`, so the bar is **0.0391% loose**. It was pre-registered *before* the
  row ran; silently retightening it is goalpost-moving, and 0.039% cannot flip a verdict whose
  other edge sits 3.2× away. **Recorded, not moved** — that is the honest call, and it is now in
  the source comment.
- **`ddm_ps1u_*.py` (2 sites) left unchanged.** That lane is CLOSED TERMINAL and harvested; the
  value is what was actually run, so changing it would rewrite a dispatched row's provenance.

## 6. Why the cure did not stop my respawn — two independent causes

The 08-16 registry row lists `charter:ddm_pu3_uncapped_pose_solve` in `propagated_into`. I was
spawned again the next day, unchanged. Measured:

**(a) COVERAGE — mine to fix, and fixed.** The 08-16 row's `claim_patterns` are all forms of
`13-23%`. My charter's *same sentence* also asserts the CAP and the MISSING CONVERGENCE TEST, and
neither had any row. Running the lint on my own charter text before my change: **1 warning** for
three dead premises. New row `pose_gn_cap_2_to_3_relinearizations_no_convergence_test` appended to
the standing store `.omx/research/falsified_premise_registry.jsonl` (13 rows). After:
**2 warnings**, and a control charter containing *"pairs 13-23"* and *"capped at 8 iterations"*
still draws **0** — the noise guard the prior arm's round-1 review demanded.

**(b) REACHABILITY — already MAIN's, not duplicated.** All three lint legs live only in
`tools/codex_arm_queue.py`; Codex is walled until Aug 20, so arms spawn through the Agent tool,
whose only `PreToolUse` hook guards model routing. MAIN registered this hours before me as
`charter_lint_is_spawn_path_conditional_20260817` **with a named cure**. I confirm it with the
receipt it lacked — **a live post-cure recurrence, me** — and stop there.

**The general shape, worth naming:** a claim-level registry keyed by phrasing catches the phrase it
was written for. My charter carried one premise in three grammatical costumes and the cure saw one.
**A cure sized to the instance that motivated it is not sized to the genus.**

## 7. Verdict

> **No measured carrier-addressable pose headroom exists on the shipping object.**
> `verdict_scope: **formulation**` — frame-0 carrier-side pose actuation on the cp135→hv1 vehicle.

Six *distinct* mechanisms are negative (relinearization budget · rank-6 basis to convergence ·
linear frame-0 overlays · ps135b carrier · ps1u uncapped solve · ra3 trust-regioned re-fit). I
checked they do not share one defect, so this is genuine family convergence, not
`[[same_defect_negatives_masquerade_as_family_convergence_20260805]]`.

**NOT a paradigm verdict.** The CUDA-side pose floor is genuinely unmeasured, and
joint-descent-through-the-shipping-receiver — the mechanism that *built* the incumbent carrier in
PR130 — has never been re-run on this vehicle.

**No sealed T4 fire-order is emitted.** My charter conditions one on "if it clears." It does not
clear: the mechanism it names was already dispatched and refused at **+1.686e-02 S**, and the
ceiling caps the whole axis at 86.43% of the gap. Emitting a fire-order would be spending MAIN's
budget on a re-run of a refused row.

## 8. Fire-orders for MAIN (all free, none dispatch)

1. **Land MAIN's own registered cure** — feed `tool_input['prompt']` through the three lint legs
   inside `agent_model_routing_guard_hook`, fail-open, plus a test that the Agent path and the
   codex path return the SAME verdict on the same charter. My respawn is the receipt that the gap
   is live, not theoretical.
2. **Re-lint the two charters still carrying the dead premise** —
   `ddm_b2e_train_for_editability_burn2_charter_20260816.md` and
   `ddm_ns1_negative_signal_audit_and_missing_patterns_20260816.md`. They now draw the CAP warning
   too, not only the percentage.
3. **Adopt the §4 break-even closed form** as the pose screen for any future candidate, adjudicated
   on `score_recomputed_from_components` vs `0.15959729295498598` — never on a `d_pose` restatement.
4. **Aim the next pose unit at the one unmeasured thing or not at all**: the CUDA-side floor
   (needs CUDA; do not fund before the constant scope-fix propagates). Otherwise **reassign** — on
   the arithmetic, **seg (0.029611 = 3.09× the gap)** and **rate (0.121692 = 12.7× the gap)** are
   where the gap is closeable.
5. **DO NOT re-open**: the relinearization cap (closed 5× now), the linear frame-0 overlay family
   (pk4, formulation), the trust-regioned carrier re-fit (ra3, family).

## 9. My own round-1 adversarial review

1. **Am I re-reporting the 08-16 pu3 memo?** In §1–§4, substantially — and I say so rather than
   dressing it up. I re-derived every number independently (which *confirmed* pu3 and pv1 exactly,
   and reproduced pk3's 0.497% as a third cross-check), but the conclusions are theirs. My own
   contributions are §5 (the constant fix, *executed* — both prior arms only raised it), §6(a) (the
   coverage gap, found and closed), and the recurrence receipt.
2. **Is the coverage claim solid?** Yes, and it is falsifiable: the lint went 1 → 2 warnings on
   fixed input, and 0 on a control. That is a measured before/after, not an assertion.
3. **Did I overstate the constant fix?** It is worth 3.4e-06 S — **0.035% of the gap**. It is not a
   score move and I do not present it as one. Its value is that a live bar stops being computed
   from another archive's number.
4. **Is my registry row orphan-risk?** Partly. It fires today, but its reach depends on cause (b),
   which is MAIN's to land. A row in a store read only by a bypassed path is worth less than it
   looks — I state that rather than claim the gap is closed.
5. **Should I have run a measurement?** I decided no and gave the reason (§2). The risk is that I
   am rationalising inaction. The check I applied: what would n=120 change if it also returns
   0/120? Nothing — the cap is *absent from source*, so pairs cannot stop on it. The source read
   dominates any sampling result.
6. **Unverified:** whether the two 08-16 charters were spawned through `codex_arm_queue` at all (if
   hand-spawned, the lint never ran); and I did not annotate `terminal_pose_gn.py:520`'s citation of
   the falsified figure — it is a comment on a correct decision, and I chose not to churn it.

## NEXT_IF_RESUMED

Fire-order 1 (MAIN's hook cure) is the only item with compounding value; everything else on this
axis is closed or arithmetic. **The pose axis needs no further local analysis.** On the numbers,
pose caps at 86.43% of the gap with six negatives against it, while seg and rate together are
15.8× the gap — that is where a unit aimed at 0.15 belongs.
