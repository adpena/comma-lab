# ddm_rx1 — the rv1 reactivation rows: 5 of 8 were already discharged; the ledger could not say so

**POINTER HONESTY FIRST.** `0.1910828242 [contest-CPU]` **UNMOVED**; competitive bar `0.172141`
[external official contest-CUDA]. Nothing in this unit is a score claim: no scorer job, no training,
no dispatch. Every number quoted below is re-derived from a committed artifact or a primary file and
carries its own axis tag. `research_only=true` · `score_claim=false` · `promotion_eligible=false` ·
`[macOS-CPU advisory]`. `[no-triality]` (apparatus + this graph leg) `[p0-ledger-ok]`.

## VERDICT (one line)

**The arm's premise was false in the direction that matters: 5 of the 8 rv1 reactivation rows had
ALREADY fired — R8 on 07-28, R1 and R7 on 07-29, R4 and R2 on 07-30 — and the ledger reported all 8
as pending because `reactivated` was a hardcoded `False` literal with no writer anywhere in the
module.** The two rows this arm was chartered to fire at $0 (R7, R4) were measured two and three days
ago, and one of them (R4→gr1) produced the `cell_drop50` base that is inside the live v4c/v4d
candidate. The fix is the write path: `reactivated` now resolves from committed, content-hashed
evidence on disk and fails closed to `False`. **Nothing was re-run; nothing needed to be.**

## §1 The defect (re-derived, not recalled)

`src/tac/ddm_costate_organ.py`, the co8 fold, pre-rx1:

```python
"reactivated": False,  # PROPOSAL until its named measurement lands
```

A literal inside a dict comprehension. `grep -n reactivat` over the whole module returns no writer —
the field is not a state, it is a constant. It therefore cannot record the event it names, which is
the **default-off orphan class** applied to a ledger field, and NO-FAKE forbidden class #2 on the
test side: `test_ddm_co8_costate_organ_round8.py:76` asserted `row["reactivated"] is False` **for
every row**, so the suite verified the constant and froze the bug in place. That test would have
passed unchanged with the function body replaced by its markers.

**Blast radius, measured.** `tools/costate_digest.py` calls `build_live_ddm_costate` + `digest_lines`
at SessionStart. The emitted line derived its headline from `by_status`, which is rv1's own
*schedulability* field frozen at 2026-07-28 — not a state:

```
DDM-rv1[consumed]: reactivations=8 (now=$0x2 armed=1 post-burn=1) closed=12 …
```

`now=$0x2` = R7 and R4. Every session from 07-29 onward was told to go do two free measurements that
had already been done. This arm is the receipt: it was chartered off that line.

## §2 The re-grade — all 8 rows, evidence-derived

`grade`: **0** = discharged (named measurement landed) · **2** = built + wired, never fired ·
**3** = designed but UNBUILT (debt). `LANDED` means the measurement landed — **never** that the lever
paid; two of the five landed as measured negatives against their own pre-registered GO tests.

| # | row | grade | state | owner | what is missing | what closes it | fire order |
|---|---|---|---|---|---|---|---|
| 1 | **R1** terminal-band discrete search | 0 | LANDED 07-29 | pb1 | nothing for R1 itself | — | done |
| 2 | **R7** token-stream coder race | 0 | LANDED 07-29 | r7 | nothing for R7 itself | — | done |
| 3 | **R4** token-granularity probe | 0 | LANDED 07-30 (as gr1) | gr1 | nothing for R4 itself | — | done |
| 4 | **R2** correction-stream band re-price | 0 | LANDED 07-30, **NO-GO** | co9/ea1 | nothing — it is refuted | — | closed |
| 5 | **R6** Lane-channel in-training entrants | 2 | OPEN, in flight | burn-4 (b4s) | the burn-4 endpoint n600 row | endpoint d_seg < 0.00426407708 control ⇒ GO, else INSTANCE close | **live now** |
| 6 | **R8** solve-INIT tokens | 0 | LANDED 07-28, **ADOPTED** | sc2 | nothing — it is ON everywhere | — | done |
| 7 | **R3** directional conditioning race | **3** | OPEN, **UNBUILT** | unowned | the channel + DSL lever do not exist on tr1 | build, then a matched from-scratch ON/OFF n600 A/B | after a build charter |
| 8 | **R5** step/hosc head | 2 | OPEN, non-binding | — | nothing to act on | a periodic-activation head (none exists) | leave closed |

Per-row receipts (each re-derived by reading the artifact, not by recall):

- **R1** — `ddm_pb1_postburn_completion_20260729.md`: §2 QDBS = 49 evals, honest-axis mode, commit
  `838b5adfbc`; #400 diagonal EXPLICIT = §4 renderer leg + §5 dxi pose-polish leg. pb1's own owed
  table marks it DONE. **Caveat travels with it:** the 0.05–0.07 S prior was witness-vehicle /
  foreign-parent, not a same-parent quote. **Open sibling, not R1:** the FULL-POPULATION GN/CG seg
  solve (orphan QA03) never ran — the −0.138 ceiling is still UNMEASURED (Contrarian bound −0.046).
- **R7** — `ddm_r7_token_coder_race_20260729.md` + receipt JSON: 14 token-entropy arms
  (KT-prev1 / CTW / rANS / Bayes-mix vs Brotli-Q11 / LZMA / Huffman), every admitted row a
  materialized `R7PL` frame with exact parse-back. The stale coder negatives are **cleared off the
  token object**, but the race handed back **no free bytes at the endpoint**: solve-project endpoint
  winner SMEVR 557,238 B → 562,174 B composed; deficit **+371,840 B** to the 0.172 ceiling. The
  zero-init T2 winner (Bayes-base+Brotli-delta, 360,743 B) is a **different lineage and does not
  transfer** — R7's own FEED says so explicitly. `OWED_EG1_INTEGRATION_NOT_AN_ARCHIVE_ROW`.
- **R4** — `ddm_gr1_granularity_rerace_20260730.md`, a **superset** of the named probe (archive-
  faithful re-quant → real SMEVR bytes + realized n600 d_seg through the frozen CPU SegNet; gates
  PASS, baseline injection 0.0038892 vs `evaluate.py` 0.00389011, Δ 1.9e-6). Its own pre-registered
  GO test **FAILED at token granularity**: every candidate worse on realized seg+rate, B/flip
  0.04–0.51 all below the 1.273 water ⇒ token-granular correction STRICTLY DOMINATED
  (scope INSTANCE/FORMULATION). The same probe found the **unit**: the *cell*, not the token
  (SMEVR conditions on per-cell temporal mode). `cell_drop50` = 359,221 B @ realized n600 d_seg
  0.004310 → seg+rate 0.6702 = **−0.098** vs the 0.7685 reference, byte-closed `a6398e44`. **That
  base is consumed by v4b/v4c/v4d.** Also overturned QA11 and dominated QA07.
- **R2** — trigger fired exactly as pre-specified (burn endpoint realized d_seg **0.0038892** ∈ the
  armed **[5e-4, 1e-2]** band), then priced and **refused on its own GO test**: co9 ΔS_seg −0.001582
  = 1.15% of the −0.138 ceiling at **1.45 B/flip ≈ the 1.27 water**. `ddm_ea1_einsteinian_negative_
  audit_20260730.md` generalizes it — at an in-band base the seg residual is **WHITE**, and post-hoc
  correction streams are measured non-paying at **BOTH** base regimes (verdict_scope: FORMULATION).
- **R6** — **the named race did not run in its own burn window.** Re-derived from the sealed ticket
  `.omx/research/configs/ddm_tb1_t3_long_burn_lotto_20260728.json`: 9 levers, argv fires a SINGLE arm
  at fixed `--class-weight-lane 1.0`; the Lane-pool race exists only in the ticket's *adjudication
  caveat* ("Lane pool race … is the FIRST burn item"), never in its levers. None of the three
  entrants (band-ACTIVE / in-training dash-comb / fixed-gate lane-prior) entered. Independently
  re-found by `ddm_fh1_forces_harvest_20260731.md` A6 ("DSL lever EXISTS … default 1.0 =
  **never-fired**") and re-chartered as the burn-4 S1 fire (`class_weight_lane` 1.0→1.3).
- **R8** — `ddm_sc2_schedule_optimality_convocation_20260728.md` row 14: the named A/B ran at matched
  epoch, n600 full-confirm, `token_init_mode=solve_project` **0.009839 vs zero-init 0.013833 =
  −28.9%**, **ADOPTED**; v1/v2 formulations measured inadmissible (FORMULATION). Now ON in every
  current config including the bc1 QA24 re-burn. R8's row was stale within ~24h of rv1 being written.
- **R3** — not merely un-fired, **unbuilt**: the sealed ticket carries no CLADE-ICPE slot, and
  `grep` over `experiments/train_tr1_partition_renderer_mlx.py` finds no oriented / directional
  conditioning lever (its only "directional" surface is the sn1 Road↔Lane asymmetric loss weight — a
  different object). The vehicle changed under the row: the −48% evidence was the witness INR's, and
  CLAUDE.md already routes self-orient OFF pending a matched from-scratch A/B. **This can never be a
  $0 row.**
- **R5** — does not bind: `ddm_fh1_forces_harvest_20260731.md` L387 records no sinusoidal layers
  anywhere on the tr1 conv renderer, so there is no periodic activation for an annealed-β cure to act
  on; the hard-state selection the step family chased is already supplied by uint8-STE + the realized
  A1 gate (RACED as `tr1_token_quant_L16_round`). Its Lane-nucleation trigger is now owned by R6.

## §3 The operating-point lens — applied, and it reactivates nothing

The charter's binding question: the burn decomposition shows steady-state descent exhausted
(61 training intervals sum to −68.6% of net descent; the 2 window-RESTART intervals carry 168.6%), so
**every negative recorded as "flat / no descent" is suspect of having been killed at the wrong
operating point.** Applied row by row, honestly:

- **R7** — its negatives were *object*-scope ("coder dominated on int8 HNeRV weights / PR101
  symbols"), not descent-scope. Not an operating-point false negative. Already re-scoped by measurement.
- **R4** — the lens **applies and already paid**: the family was not wrong, the **unit** was. Token
  granularity is dominated; the cell is the coding+coarsening unit. That is a wrong-coordinates
  finding, the sister of the campaign's own "wall = wrong-coords" law, and it is worth −0.098.
- **R2** — the lens **confirms the negative rather than overturning it.** ea1's mechanism *is* the
  operating-point explanation: at an in-band (converged) base the seg residual is WHITE, so a
  post-hoc perturbation has no structure to exploit. Measured at **both** base regimes, and
  independently corroborated today by the ERF-collateral law (post-hoc injection is net-worse even
  with perfect GT, +0.30 S). Three independent confirmations; R2 stays refuted at FORMULATION scope.
- **R6** — its three negatives were POST-HOC or guard-tainted, so the lens does apply — and the
  in-training re-entry is exactly what burn-4 is firing. Already owned; no new duty.
- **R3** — the lens applies *most* sharply here (owed16v2 compared at matched trained cells, i.e. a
  steady-state comparison), but the row is blocked on a **build**, not on an operating point.
- **R1 / R5 / R8** — not descent-scope negatives; the lens is silent.

**Honest conclusion: the operating-point lens resurrects zero $0 rows.** The row it most implicates
(R3) is grade-3 debt. Saying so is the deliverable; padding it into a fired probe would be the fake.

## §4 The fix (class, not instance)

`src/tac/ddm_costate_organ.py`:

- `Rv1ReactivationSpec` gains `result_glob` / `landed_disposition` / `open_state` / `open_reason` /
  `charter_glob`. The fold resolves each row against `.omx/research`: a matching committed **result**
  artifact ⇒ `reactivated=True`, `duty_state=LANDED`, `evidence=[{path, sha256}]`; otherwise the
  declared open state, with a **charter** pointer where one exists (R6). **Fails closed to `False`.**
- `reactivated=True` is defined to mean **strictly** "the named measurement landed" — never "the
  lever paid". The outcome, including measured NO-GOs, lives in `disposition`; the table `boundary`
  string says this so no downstream reader can launder it.
- `counts` gains `landed` / `open` / `by_duty_state`. The digest line now reads off the **live** split
  instead of rv1's frozen schedulability field:

```
DDM-rv1[consumed]: reactivations=8 landed=5[R1,R7,R4,R2,R8] open=3
  R6=OPEN_IN_FLIGHT_CHARTERED R3=OPEN_LEVER_NOT_BUILT_ON_LIVE_VEHICLE
  R5=OPEN_NON_BINDING_ON_LIVE_VEHICLE closed=12 corrections=2
  landed=measurement-landed-NOT-lever-paid (evidence-derived, read disposition)
```

Tests (`test_ddm_co8_costate_organ_round8.py`): the constant-assert is replaced by **behavioural,
mutation-resistant** assertions in both directions — `reactivated` must equal evidence presence and
must agree with `duty_state`; a repo with the memo but no result artifacts must resolve **all 8
False** (fail-closed); the five landed rows are pinned to the artifact prefix that discharged each.
**Mutation-proven, not claimed** — both mutants were executed: `landed = True` kills 3 tests,
`landed = False` (the pre-rx1 behaviour) kills the landed-set test; the old suite killed neither.
45 pass across co7+co8; `ruff` clean. Three pre-existing failures elsewhere in the costate suite (`test_costate_digest_ncde`,
`test_ddm_co9…open_gate_ownership_scan`, `test_witness_control_costate::test_no_actuation_capability`)
were verified failing at HEAD before this change and are untouched by it.

## §5 What is owed (ranked by axis weight, not by rv1 rank)

Forest: own-vehicle S ≈ **0.964** (v4d) vs bar 0.172141 — 5.60× remains; axis split **pose ~1.24 ·
seg 0.431 · rate 0.239**.

1. **QA03 full-population GN/CG seg solve** — R1's open sibling, never ran; the −0.138 ceiling is
   unmeasured (book bound −0.046 = 14% of it). **Seg axis.** Needs a scorer slot; queue it, do not
   take the slot from the live eval.
2. **R6 burn-4 endpoint row** — in flight; closes on `d_seg < 0.00426407708`. **Seg axis.** No action.
3. **R7's +371,840 B endpoint deficit** — the coder race is done and honest; the remaining rate work
   is the waterfill rung + granularity re-race already on the board. **Rate axis (0.239).**
4. **R3 build charter** — grade-3 debt, unowned. Rank it against pose/seg, not against its rv1 rank:
   it is a seg-conditioning channel on a 0.431 axis requiring a build, which puts it **below** every
   pose item at the current operating point.

**Scope of this unit:** the eight reactivation rows only. The twelve rv1 non-reactivation rows were
not re-graded here; X5 (chroma <2px reclassified as a pose-safe exploit) touches the largest axis and
is flagged for whoever owns the terminal pose solve.

## §6 Ledger hygiene note

The rv1 rows are **not** carried in the consolidated deferral queue ledger
(`ddm_deferral_queue_ledger_20260729.md`) — its `r7` rows are the r7 *memo's* QA items, a different
namespace. Per the defer-at-source rule that is a second orphan surface; the organ now being
evidence-derived closes the practical harm (the SessionStart line no longer misdirects), but folding
the 3 open rows into the canonical ledger is left to that ledger's owner rather than edited from here
while other arms are live in it.
