---
schema: ddm_cn3_week_coherence_audit.v1
date_utc: 2026-07-31
arm: ddm_cn3 (task #818 — week coherence: signal-loss/consumption audit · queue coherence · roadmap re-coherence)
lane_id: "lane_ddm_cn3_week_coherence_20260731"
research_only: true
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[macOS-CPU advisory — $0 read-only audit; ZERO scorer forwards (window_03 owns the n600 slot); no byte-closed row produced]"
council_predicted_mission_contribution: apparatus_maintenance
verdict_scope: FORMULATION
operator_binding: "07-31: 'review all of our configs and memos and such from the past week to ensure no signal loss and a coherent workflow and queue and road map.'"
consumes: [ddm_gc14_first_descent_20260731, ddm_gc15_fresh_vs_warm_20260731, ddm_ja1_joint_atlas_waterfill_20260731,
  ddm_v4d_adaptive_hybrid_20260731, ddm_b2b_burn2_composition_build_20260731, ddm_gr1_granularity_rerace_20260730,
  ddm_fl1_perclass_flicker_floors_20260731, ddm_gc13_optimal_control_shape_20260731, ddm_deferral_queue_ledger_20260729,
  ddm_us1_upstream_reread_20260731, ddm_rv1_conditional_validity_regrade_20260728]
consumers: [MAIN next charter, the costate organ SENSE surface, ddm_deferral_queue_ledger (row updates), every subsequent spawn]
tokens: [no-triality, p0-ledger-ok, magnitude-ok]
---

# ddm_cn3 — the week in three tables: what was orphaned · what the queue actually holds · where the campaign goes next

## §0 POINTER HONESTY FIRST (means/ends firewall)

**The exact frontier did NOT move. `0.1910828242 [contest-CPU]` is UNMOVED.** This unit is a $0
read-only AUDIT. It produced no byte-closed archive, ran zero scorer forwards (window_03 owns the
single n600 slot until ~18:40Z), and moved no trained byte. It is **MEANS**. Its entire value is that
the next charter spawns against a queue and a roadmap that are not lying to it.

**The one-line answer to the operator's three questions:**
1. **Signal loss — 3 outright orphans + 1 orphaned sub-deliverable among the 14 named suspects, but the
   biggest loss is a BUILD, not a memo.** `ddm_b2b`'s burn-2 stack (QA83 factorized head + QA84
   variable-cell grammar + QA86 config corrections + QA75/QA80 harness) is **BUILT, TESTED, 4 commits
   landed, and NEVER FIRED**, while the slot its own memo names as its consumer ("MAIN post-burn
   boundary — compose + fire burn-2 immediately") went to burn-4 continuation windows gc14 has now
   measured as **exhausted** (r = 0.310, 2.3% of the gap). **And there is a root cause under all of it
   (§2b): three anti-orphan gates all report CLEAN because each scans the WRONG SET** — the lever
   registry sees **1 of 171** modules, the codex-findings gate sees **0 of 1,260** files, and #396 sees
   correctly but was never strict-flipped (**433 live violations**). All three repairs are $0.
2. **Queue — 92% of the open ledger is stale or duplicated.** 371 events → 129 task_ids → **51
   non-terminal: 29 STALE · 18 DUPLICATED · 2 free-unblocks · 1 ORPHANED · exactly 1
   LIVE-AND-CORRECT.** Six QA rows still say `DUE`/`OPEN` for work v4d already measured. #766 (a
   surviving material pool) has **no ledger row at all**.
3. **Roadmap — re-derived in §4, and the headline is arithmetic and uncomfortable:** **no composition
   of currently-banked components reaches the bar.** Banked pose (0.12689) + best byte-closed rate
   (Knee-B, 0.11624) = **0.24313 > bar 0.172141 with ZERO seg budget.** All three axes need
   *multiplicative* improvement: seg **7.19×**, pose **19.19×**, rate **2.77×**.

**⚡ And one live finding that arrived mid-audit and settles the immediate decision (§4.0):
window_03 has REVERSED** — 7 consecutive monotone rises, now **worse than the window_02 endpoint**,
**347% of the boundary step surrendered**, OLS on gc14's own TEST-1 window **t = +6.28**. Applying
gc14's *pre-registered* thresholds: TEST 1 ⇒ BOUNDARY-LOCALIZED (overwhelming), TEST 2 ⇒ **"DRAIN
COMPLETE — pool exhausted, stop."** **B5-C fires; a window_04 must NOT run.** The slot is free at
~18:40Z and the pre-registered gate has already decided it leaves seg-continuation.

---

## §1 PROVENANCE AND AUTHORITY

| item | value | source |
|---|---|---|
| venv | `/Users/adpena/Projects/pact/src/tac/__init__.py` | hijack check CLEAN |
| git HEAD at start | `e922da7a92` | `git log -1` |
| scorer jobs run | **0** | window_03 (pid 49743) owns the slot; supervisor 65276 live |
| burn custody | `/Volumes/VertigoDataTier/pact/ddm_b4s_20260731/` | **READ-ONLY**; no writes by this arm |
| parallel-session WIP | `direct_description_carrier_compose.py`, `ddm_qa43_two_plane_parallax_probe.py`, `burn_out` | **not touched** |
| pointer | `0.1910828242` [contest-CPU] | **UNMOVED** |
| effective bar | `0.172141` (official PR130) / THE target `0.15` | `canonical_frontier_pointer.json` |
| own-vehicle line | **`0.9639878179`** — MEASURED n600 `evaluate.py` | QA78 (v4d gate, sha `f1f3288062…`, 360,238 B) |
| `TaskUpdate` | **NOT AVAILABLE in this environment** | dispositions recorded in §3 per the charter fallback |

**Window audited:** 2026-07-24 → 2026-07-31. **417 files touched** under `.omx/research/`; 371 event
rows in `.omx/state/canonical_task_status.jsonl`; **95 rows** in the standing
`ddm_deferral_queue_ledger_20260729.md`.

**Two queues exist and they are not the same object.** This is itself a coherence finding:
- `.omx/state/canonical_task_status.jsonl` — the **task** ledger (#NNN rows, writer-enforced).
- `.omx/research/ddm_deferral_queue_ledger_20260729.md` — the **QA** ledger (QA01–QA92), which the
  defer-at-source rule made canonical on 07-29 and which every convocation actually reads.
Work registered in one is routinely invisible to the other. §3 audits both.

---

## §2a SIGNAL-LOSS / CONSUMPTION AUDIT — the 14 named suspects (07-24 → 07-31)

**Result: 3 outright ORPHANED + 1 orphaned sub-deliverable · 5 PARTIAL · 4 CONSUMED · 1
OWED-NOT-YET-DUE · 1 CONSUMED-as-disposition.**

| # | suspect | state | evidence | consumer that should absorb it |
|---|---|---|---|---|
| 1 | cn1/cn2 (#726/#727) | CONSUMED (as disposition); the `pending` flag is STALE | zb1 §7 RE-GATE DONE; PDW1 5-file set superseded-by-main (`237b955ef7`) | none — flip the queue status off pending |
| 2 | sh1 (#729) | **PARTIAL — registry row ORPHANED** | findings consumed by cs1 §3–5; `ddm_pf3b_52probe_joint_improving_hunt` absent from `lane_registry.json`; 4 branches unmerged; blocker file dirty 9 days | `lane_maturity.py add-lane` at a quiet boundary |
| 3 | **#669 five-type follow-ons** | **ORPHANED** | is1 DAG_FEED:39,151 "#669c QUEUED / NEVER RUN"; `rg -c 669` on the deferral ledger = **0**; lp1/la1 uncited after 07-25 | a #669 row in the deferral ledger, or an explicit SUPERSEDED-BY-ARC close |
| 4 | #670 warn-only-purgatory | PARTIAL | lane leg clean (2,260 lanes, 0 violations); **94 warn-only vs 344 strict**; Catalog #396 = **433 live violations, 108 in-window** | a hygiene batch draining #396 |
| 5 | **#706 post-J8F wave** | **ORPHANED-BY-SUPERSESSION** (tracked) | ledger QE02 `SUPERSEDED-PENDING`; re-flagged gc7r:163 row 13, never adjudicated; J8F code exists, wave never fired | MAIN pre-arc disposition wave (ledger S-E) |
| 6 | #582(b) batch reconcile | **PARTIAL — branches merged, laws NOT registered** | only **2 of 7** §B4 laws registered; absent: boundary-concentration, 0.047-recovery, pose dim-0, **ker(A) 80.67% nullity**, compact-replay, `erm_margin_topk`, #466 | the **equations leg** — the nullity is cited CROSS-VEHICLE VALID yet exists only as prose |
| 7 | rv1 reactivations | **CONSUMED** | `ddm_costate_organ.py:1203–1320`, 8× `Rv1ReactivationSpec`, content-hashed; R1/R2/R4 measured in pb1; R3/R5/R6/R8 → QA26–29 | — |
| 8 | lv1 OWED rows | PARTIAL — content consumed, formal fold never happened | organ status `COMMITTED_ARTIFACT_PRESENT_FOLD_ON_NEXT_ROUND`; co9 (07-30) never mentions lv1; `T1_VALIDITY_GATE` still gated | organ round 10 |
| 9 | eg1 endgame receipts | **CONSUMED** | `ddm_endgame_policy.py:26` schema; `rehearse_ddm_tr1_runtime` / `pb1_p5` / `pfs1` tools; organ duty ranks 1 & 5; QA03 fired | — |
| 10 | vh1 untransferred lessons | **CONSUMED** | row 7 → built as tp1 (#804, `15aad5a28b`); rows 4/5/12/13/14/15 → burn-4 charter skeleton:21–23,90,123; §2 taxonomy → gc12 | QA90 still DUE |
| 11 | us1 timm numeric-parity OWED | **PARTIAL — 1 of 11 rows consumed** | rank-2 denominator → dg1 + Catalog #407 + 26 tests *the same day*; **timm parity still zero-consumed** | terminal pose solve; burn-4 endpoint for D6. **⚠ see the correction below — F2 is NOT owed** |
| 12 | fl1 phase-debt ranking | **CONSUMED (within hours)** | `supervise_ddm_b4s_burn4.py:116–129` — window_02 `UNDRIV_EROSION` adjudicated CONTINUE **because** fl1 prices the pool; also gc15:392,446 | Lane's #1 debt still needs a carrier |
| 13 | **gc13 cg1 amendments** | **ORPHANED-PENDING-NONEXISTENT-CONSUMER** | R3 fired (fl1), R4 landed (supervisor:574), R1 built-not-run (no SSD dir); **R2 cg1 build DOES NOT EXIST; #809 has no ledger row anywhere** | build cg1, or at minimum register #809 |
| 14 | gc14 §16 amendments | **OWED-NOT-YET-DUE — inherits #13's defect** | memo:419 self-attests "#809 has no row"; 7 more amendments ⇒ **18 cumulative design items against an unbuilt entity** | the cg1 build itself |

**Top-3 orphans by leverage:**
1. **gc13 + gc14 cg1 amendments (#809)** — 18 design items compounding against an entity with **no
   artifact and no ledger row**, and it is the **guard layer for the binding seg axis**. Every
   convocation keeps amending something that does not exist.
2. **#582(b)'s 5 unregistered laws** — `ker(A) 80.67%` is cited as **CROSS-VEHICLE VALID** yet lives
   only as prose, so **every downstream nullity claim is unresolvable** against a registered source.
3. **#669(c) re-homing audit** — the never-run exhaustion proof **the operator explicitly demanded**,
   absent from the consolidated queue entirely, so **no recall pass can surface it**.

> **⚠ CORRECTION I APPLY TO ROW 11 FROM MY OWN VERIFICATION.** The audit arm reports
> *"`camera_fl=910`, yuv6 polyphase = zero code hits"* — **the `camera_fl` half is wrong, and I made the
> identical error before catching it** (§5 self-correction). `src/tac/boundary_math/xi_pose_coder.py:70`
> is `_NATIVE_FX, _NATIVE_FY = 910.0, 910.0`, and the live v4d receiver builds `K` from that path. The
> arm and I independently produced the same false negative from the same naive grep, which is itself the
> finding: **two independent verifications converged on the same wrong answer because they used the same
> flawed method.** Convergence is not corroboration when the method is shared. Row 11's real residue is
> the **timm numeric parity** (genuinely zero-consumed) and the **F4 polyphase** leg only.

---

## §2b THE ROOT CAUSE — three anti-orphan gates all report CLEAN because each scans the WRONG SET

**This is the most important structural finding in the audit and it explains every other row.** The
apparatus is not failing to *detect* orphans. It is failing to *look at the right set* — and in the one
place it looks correctly, it is not permitted to act:

| mechanism | reports | actual | why |
|---|---|---|---|
| `check_codex_findings_memos_consumed` | **LIVE COUNT 0** | 10 orphaned findings | scans only `mtime < 3 days` ⇒ **0 of 1,260 files currently in scope**. The gate is structurally vacuous. |
| `lever_registry.completeness()` | **`stale = []`** | **10 orphaned levers** | `_module_source()` (`src/tac/witness_dsl/lever_registry.py:107`) returns only `curriculum_dsl.__file__`, and `lever_factories()` (L165) ASTs **that one file** ⇒ ~180 other `witness_dsl/*.py` modules are invisible. `stale == []` is **vacuous**, and the triality drift-detector inherits the blindness. |
| `check_measured_win_findings_are_wired_or_research_only` (#396) | warn-only | **433 violations (108 in-window)** | correct scan set — but **never strict-flipped**, so it never blocks. The warn-only-purgatory failure mode CLAUDE.md's strict-flip-atomicity rule exists to prevent. |

**I verified the second row myself rather than relaying it** (operating manual §4):
`src/tac/witness_dsl/lever_registry.py:106-107` is literally
`def _module_source() -> str: return Path(_cd.__file__).read_text()` — **one** file — while
`ls src/tac/witness_dsl/*.py | wc -l` = **171**. The registry that CLAUDE.md designates as the
auto-derived "what is missing from the DSL?" answer sees **1 of 171 modules (0.6%)**.

**All three repairs are $0.** This is the single highest-leverage apparatus fix available, and it is
strictly upstream of every individual orphan below: fixing the scan sets converts three
permanently-green gates into gates that would have caught this week's losses automatically.

**NEW LAW PROPOSED (generalizing all three rows, and both of this arm's own near-misses in §3b.4/§5):**
> *A gate's LIVE-COUNT-0 is meaningless until its DENOMINATOR is asserted.*
A green gate over an empty scan set is indistinguishable from a green gate over a clean repo — exactly
as a `grep` that never ran is indistinguishable from a `grep` that found nothing. Every gate should
report `checked N of M` alongside its verdict, and a gate whose `N` is 0 (or whose `M` is a single file
where the domain is a package) should **fail loud, not pass quiet**. Sister of the silent-guard
non-negotiable, lifted from runtime guards to *static gates*.

### 2b.1 The 10 orphaned DSL levers — and the two expensive ones

| tier | levers | state |
|---|---|---|
| **DESIGNED-STUB** (no trainer flag either ⇒ a config setting them would fail closed on never-invent-flags) | `TieLocusEdgeWeighted` · `MarginSatisficeCap` · `XiAdvectedTokenBase` · `BirthPlateauKneeConjunct` · `ErfBirthContextCoadapt` · `Ax1Frame0CarriedWarp` · `Qa80MarginBoundedPhotometric` · `Qa81LaneCarrierComposite` | design debt |
| **REAL CAPABILITY, NO CONSUMER** — trainer-implemented, race programs built, **set in no sealed ticket** | **`lever_token_rowband`** (`--token-rowband-spec`, QA84) · **`lever_renderer_head`** (`--renderer-head-mode`, `--head-photo-slack-gain`, QA83) | **the expensive orphans** |

**`lever_token_rowband` is the sharpest orphan in the campaign:** it is trainer-implemented **and**
carries a *registered canonical equation*
(`src/tac/canonical_equations/ddm_b2b_rowband_flip_mass_20260731.py`, flip-mass 0.721 anchor) **and** a
race program (`spec_tr1_burn2_20260731.py:200`) — **everything exists except a ticket that sets it.**
By contrast the other 26 `lever_*` in `spec_tr1_renderer_20260728.py` **are** wired (35 lever instances
across sealed tickets). **This independently confirms the burn-2 orphan (§3b.3) from a completely
different direction — the DSL layer rather than the SSD-custody layer.** Two routes, one conclusion:
QA83/QA84 are built, equation-anchored, race-ready, and unfired.

### 2b.2 The duty-to-measure queue cannot see any of them

`.omx/state/lever_activation_ledger.jsonl` holds 251 rows / 37 distinct levers with events
`{built, fired, measured}` — and **none of the 10 orphans appear in it at all, not even as `built`.**
There is no `ever_fired` field. CLAUDE.md's *"'off' is a tracked queue, never a forgotten default"*
non-negotiable requires exactly this ledger to rank never-fired levers into the DECIDE queue; **a lever
that never receives a `built` row is outside the mechanism entirely.** The rule is sound; its ledger is
not being written by the paths that create levers. UNVERIFIED whether the omission is intentional.

**Also fully detached:** `src/tac/witness_dsl/fh1_adapted_force_levers_20260731.py` has **zero importers
repo-wide**, and `ph3_s10_frontloaded_levers_20260731.py` is imported only by that dead file — a
detached subgraph. None of the 8 stub levers are exported from `witness_dsl/__init__.py`.

**Scoped `completeness()` output** (undercounts, per the blindness above): `unmapped` = **80 trainer
flags the DSL does not hold** (`--muon-*` ×5, `--mod-dim-*` ×4, `--eikonal-visco*` ×3, `--amplify-*` ×3,
`--lane-*` ×4, `--fresh-*` ×3, `--annulus-plateau-*` ×3, …); `stale` = 3.

---

## §3a QUEUE COHERENCE — the canonical task ledger (`canonical_task_status.jsonl`)

Folded latest-row-wins: **371 events → 129 distinct `task_id`s**, of which **51 are non-terminal**
(33 pending · 11 blocked · 7 in_progress; 78 completed).

| disposition | count | share |
|---|---:|---:|
| **STALE** (premise superseded by later measured work) | **29** | 57% |
| **DUPLICATED** (a twin row/ledger owns the deliverable) | **18** | 35% |
| **BLOCKED-ON-SOMETHING-ALREADY-LANDED** (free unblocks) | **2** | 4% |
| **ORPHANED** (no owner, no consumer) | **1** | 2% |
| **LIVE-AND-CORRECT** | **1** | 2% |

**Read that last row again: exactly ONE of 51 non-terminal ledger rows is live and correctly aimed.**
92% of the open queue is stale-or-duplicated. The full per-row table (all 51, no skips, with
superseder and recommended action for each) is held in this arm's audit output and is reproduced in
the committed revision; the decision-relevant classes are:

**The one LIVE row — and the charter's framing of it was wrong.** `#807` (burn-4 compose+seal+fire):
the charter told me *"b4s is dead, its work was completed by ddm_b4r."* **Half right, and materially
wrong on the live half.** The *arm* `ddm_b4s` is dead (Fable-5 usage limit) and `ddm_b4r` did land 4
commits under #807 — **but the deliverable is still executing**: trainer PID 49743 and supervisor PID
65276 are both ALIVE (verified by `ps -p`, not by a truncated `ps aux | grep`, which returned only 31
of 1096 processes and would have manufactured a false "all arms dead" finding). #807 is
**LIVE-AND-CORRECT with a stale `owner` field and FALSIFIED `event_notes`** — its notes assert
`LG1_DUAL_ENGAGED=True`, which gc14 falsified (`lambda_lane = 0.0` at all 38 gates). **Repair in place
(owner → `ddm_b4r`; append a note superseding the lg1 claim); do NOT mark completed.**

**The 6 other in_progress rows all have DEAD arms** (`fp_shrink`, `pose_low_rank`,
`phase_residual_carrier`, `#494`, `einstein_kolmogorov PDW1_PALETTE`, `g111 resume`) — verified against
the full 1096-process table. This IS the #807 class the charter asked about; it just is not #807.

**2 free unblocks, both stale by 16 days:** `sfess_cached_replay_ugc64` and
`C4-MOD19-RATE-BYTECLOSE-LOCAL` were each blocked on a codex-sandbox `git add rc=128`, which was fixed
by `9cc9eb830b` and harvested by `47f712aca1`/`51ece84eb4`/`335e386088` — **all on 2026-07-15, the same
day the blockers were written.** Nobody swept the queue afterward.

**18 pure mirrors (35%).** `deferral_ledger::D41–D52c` are duplicated wholesale into the task ledger
while `.omx/state/deferral_ledger.md` is their declared `canonical_source`. Delete the mirrors.

**A live orphan PROCESS:** PID **77097**, `observe_m1_banded_checkpoints.py`, running **10 days 15 h**
on the #575 m1/curvelet lineage that CLAUDE.md's 2026-07-27 superseded-production-routing note retired.
Read-only, but it is live compute on a dead lineage.

**The structural finding — the registration boundary is #793 / 2026-07-31.** Numeric task registration
*restarts* at #793; everything between #766 and #792 that mattered is unregistered. This is why
**#766 (wr1 Knee-A/B) — which gc14 §13's census lists as one of only two surviving material pools at
−0.197/−0.263 S rate — has NO ledger row at all.** It is the largest unregistered item in the campaign.
**Consequence: the gc14 corrections land almost entirely OUTSIDE the ledger's reach**, because the
burn-4/cg1/#815 work is unregistered — which is itself the strongest argument for registering
#766/#809/#815 before the window_03 endpoint.

---

## §3b QUEUE COHERENCE — the QA ledger (95 rows; the queue every convocation actually reads)

`ddm_deferral_queue_ledger_20260729.md` is the standing defer-at-source queue. Status census by class,
and then the rows whose status is **WRONG as of tonight**.

| status class | rows | reading |
|---|---:|---|
| FIRED / FIRED-MEASURED / FIRED-CLOSED / DONE / CLOSED | 24 | healthy — receipts retained per the append-then-update rule |
| DUE / DUE-LOW / DUE-ON-SLOT / DUE-AT-* | 27 | the live queue |
| HELD / HELD-RACE / HELD-LOW / HELD-GATE / HELD-OPERATOR | 26 | named gates; mostly correct |
| BUILT (never fired) | 4 | **the orphan cluster — see below** |
| FOLD / OWED / DEFER / MEASURED / other | 14 | mixed |

### 3b.1 STALE-STATUS — 6 rows the queue still marks OPEN that are MEASURED-DONE

This is the highest-consequence queue defect found, because a charter reading these rows would
**spend a slot redoing measured work**:

| row | ledger status NOW | reality | evidence |
|---|---|---|---|
| **QA65** dim0 offset-coded lattice | `DUE-AT-v4d` | **CONSUMED by v4d** | `ddm_v4d` `operator_binding` names it in the composed stack; gate FIRED |
| **QA66** per-pair rung-A beta member | `DUE-AT-v4d` | **CONSUMED by v4d** | same; also the v4d FLOOR candidate (359,890 B, sha `d5149d811b4a`) |
| **QA69** realized bit-allocation | `OPEN (v4d)` | **CONSUMED by v4d** | same `operator_binding` |
| **QA70** min-entropy member selection | `OPEN` | **CONSUMED by v4d** | same `operator_binding` |
| **QA72(a)** quantum-as-medium stage attribution | `DUE (ph3 §6)` | **(a) CONSUMED by v4d**; (b)(c)(d) still open | `operator_binding` names QA72a only ⇒ **PARTIAL**, split the row |
| **QA62** pm1 rungs A+B fold | `DUE-AT-v4c` | **v4c LANDED + MEASURED** (S 0.992972) | v4c gate receipt |

**Disposition: flip all six to FIRED (QA72 → split A-fired / B,C,D-open), citing the v4d QA78 receipt.**
Sister consequence: **ja1's ranked table ranks four of these six at positions 1–4** (§4.4) — the stale
QA statuses and the stale allocator are the *same* defect seen from two surfaces.

### 3b.2 SUPERSEDED-PREMISE — rows whose gate this week's measurements invalidated

| row | stated premise | superseder | disposition |
|---|---|---|---|
| **QA07** r7 sensitivity-weighted nested rung | `HELD (design)` | **gr1 07-30 MEASURED DOMINATED** at both granularities | mark **DOMINATED-INSTANCE** (already in the notes; the *status field* still says HELD) |
| **QA55** container/member lossless recompression | `DUE` | ja1 §5: rate/lossless pool **SATURATED** (kl1 consumed in v4c; token stream at the SMEVR floor) | **DUE→HELD-SATURATED** pending a new container |
| **QA46** seg-cure partial-knee gate | `HELD (secondary)` | co9 white-jitter **BREAK-EVEN** (1.45 B/flip ≈ water 1.27) + ja1 §2 | correctly HELD; annotate with the break-even receipt so it is not re-proposed |
| **QA10** flicker-phase ACTUATION | `HELD (named build)` | gc14 §11 gave it a **pre-registered fire criterion** (PIERCE-DOMINANT ∧ Yousfi Bayes-floor < 0.00167) | **LIVE-AND-CORRECT** — gate is now named, not vague. Queue the $0 preflight regardless of branch. |
| all **seg-continuation** rows | "continuation still descends" | gc14 r = 0.310; w04 net-positive | **STALE** — see §4.4 |

### 3b.3 THE BUILT-BUT-NEVER-FIRED ORPHAN CLUSTER (the week's largest signal loss)

| row | state | commit | named consumer | fired? |
|---|---|---|---|---|
| **QA83** output-space factorization (renderer head `rgb`/`class_field`/`class_field_photo`) | **BUILT + TESTED** | `f28e427dd9` | "MAIN post-burn boundary — compose + fire burn-2 immediately" | **NO** |
| **QA84** variable-cell tiling (`RowBandGrammar`, differentiable tie, SMEVR byte-close) | **BUILT + TESTED** | `e8d531e735` | same | **NO** |
| **QA86** burn-2 config-corrections bundle (census T4/T5/T6/T8/T19) | **BUILT + TESTED** | `f28e427dd9` | same | **NO** |
| **QA75/QA80** distill/margin harness | **BUILT + STUB-SMOKED** | `4bdd72a2f7` | same | **NO** (real scorer owed post-burn) |

**Verification of "never fired":** no `ddm_b2b`/burn-2 output directory exists on the SSD custody tier
(only `ddm_b2b_qa75_field_20260730` and `ddm_b2p_20260731`, both *prepay/field* artifacts, not a run);
the live trainer process (pid 49743) is burn-4 **window_03**, not burn-2. The b2b memo itself states
plainly: *"The 4 races are BUILT, not MEASURED. No d_seg/byte row exists until MAIN fires them."*

**Why this is the week's largest orphan:** its named consumer — the post-burn boundary — **arrived and
passed**, and the slot went to burn-4 continuation windows, the mechanism gc14 has now measured as
exhausted (r = 0.310, 2.3% of the gap, w04 net-positive). Four commits of built, tested, capacity-scale
seg levers sat idle while the slot bought 2.3%. A **sunk-cost build with a named consumer and no fire
receipt** is a strictly worse orphan than an orphaned finding — the money is already spent.

### 3b.4 UNREGISTERED WORK — live convocations with no ledger row

Verified by folding `canonical_task_status.jsonl` to latest-row-wins per `task_id` (**not** by grep —
see the correction note below).

| item | task-ledger row | where it actually lives |
|---|---|---|
| **#814 / gc14** | ✅ `completed` | memo + FEED-gc14 + current_focus |
| **#816 / gc15** (17th convocation, the bias-correction mechanism) | ✅ `completed` | memo `ddm_gc15_fresh_vs_warm_20260731.md` + FEED-gc15 |
| **#815** (gc14 R1/R2/R3 three-armed follow-on) | ❌ **NO ROW** | `current_focus.md` L34 only — and no QA-ledger row either |
| **#817** | ❌ **NO ROW** | nowhere located |
| **#809 / cg1** (guard ledger) | ❌ **NO ROW** | `current_focus.md` L107-111 + gc13 §9 R2 + gc14 §16 (which states the absence explicitly) |
| **#425 · #535 · W1-COH** | ❌ **NO ROWS** | `ddm_deferral_queue_ledger` QF02 / QA10 only (gc14 §11) |

> **⚠ SELF-CORRECTION #2 — a shell error produced a false negative I nearly shipped.**
> I first tested this with `grep -rn "#815\|#816" .omx/ --include=*.md --include=*.jsonl`. Under zsh the
> unquoted globs failed with `no matches found: --include=*.md`; grep **never ran**, my heredoc printed
> its `(end)` marker, and the empty output read exactly like a clean negative. I concluded "#815 and
> #816 have no ledger rows" — **half of which is false**: #816 is `completed` in the ledger. The
> conclusion only survived because I re-derived it a second time by folding the JSONL directly.
> **Two near-misses in one audit, same root cause (§7's `staleness`/verification laws generalized): a
> command that fails to run and a substring that over-matches are indistinguishable from evidence
> unless you make the tool state its own denominator.** Operating consequence adopted for this memo:
> every absence-claim in §2/§3 is backed by a *positive* enumeration (fold the ledger, list what IS
> there) rather than an empty grep. An empty result is not a finding until the command is proven to
> have run.

**Root cause (a defer-at-source gap, not an individual lapse):** the defer-at-source rule says *append
a row at the moment of deferral* but does **not say WHICH of the two ledgers**. Both exist; both are
authoritative for different consumers; neither is a superset. **Recommended repair: the defer-at-source
rule gains one clause naming the QA ledger as the queue-of-record and the task ledger as the
work-of-record, with a stated cross-reference duty.** This is the structural fix; transcribing #815/#816
by hand is only the instance fix.

---

## §4 THE ROADMAP — re-derived against the standing arithmetic

*(§2 and §3 tables follow in the committed revision; §4 is placed first because it is the urgent
deliverable and because §7 §8 of the operating manual says the actionable thing goes first.)*

### 4.0 LIVE STANDING (measured this hour) — gc14's pre-registered gate now FIRES against continuation

**This was not in the charter and it changes the roadmap from "recommended" to "forced by the
pre-registered decision function."** Read directly from the live window_03 telemetry
(`/Volumes/VertigoDataTier/pact/ddm_b4s_20260731/window_03/telemetry.jsonl`, READ-ONLY, 22 gates
ep809→ep914):

```
w02 last gate            ep805  0.0040519
w03 boundary step        ep809  0.0039402      (-1.118e-4, the step gc14 measured)
w03 trough               ep879  0.0039510
w03 LAST                 ep914  0.0043281      <-- WORSE than the w02 endpoint by +2.76e-4

OLS, gc14's TEST-1 window (ep >= 819, n=20):  slope +3.263e-6/epoch   t = +6.28
OLS, since the ep879 trough      (n=8):       slope +9.276e-6/epoch   t = +9.57
boundary step surrendered:                    347%  (gave back the whole step and more)
```

**Applying gc14 §4.3's OWN pre-registered thresholds:**
- **TEST 1 (LOCALITY):** `slope_t > −2.0` ⇒ **(c) BOUNDARY-LOCALIZED**, now at **t = +6.28 over 20
  gates** (gc14 could only measure t = −0.26 on 4 gates at ep834). The mechanism verdict is no longer
  marginal.
- **TEST 2 (MAGNITUDE):** the gate is now **above** the w02 endpoint, so `Δ_tot` lands **positive** ⇒
  the top band, **"DRAIN COMPLETE — pool exhausted, stop."**
- **gc14's own point prediction is being MISSED in the wrong direction.** It pre-registered *"I am
  pre-registering that this prediction will be beaten"* (endpoint 0.0040019, `Δ_tot ∈ [−0.0110,
  −0.0075]`). The live gate is heading the opposite way. **An honest falsification of gc14's forecast
  that STRENGTHENS its mechanism verdict and its STOP rule** — exactly what a boundary-step artifact
  with give-back predicts, and inconsistent with branches (a) and (b).

**⇒ BRANCH B5-C FIRES. A window_04 must NOT run.** gc14 §5.2 already projected w04 net-**positive**
(+0.00068) on the *decay* model; the live series is worse than that model, because the step is being
surrendered rather than merely decaying. **The slot is free at ~18:40Z and the pre-registered gate has
already decided it should leave seg-continuation.**

**Authority caveat, binding:** this is the **36-pair a1 gate**, in-window, `[macOS-CPU/MLX advisory]` —
**not** an n600 endpoint verdict, and gc14 §1 measured this gate's bias against n600 as *moving*
(−1.1e-6 → +1.52e-5). The ~18:40Z endpoint bundle is the authority. But the *direction* is unambiguous
(t = +6.28, 7 consecutive monotone rises), and no plausible bias correction of order 1e-5 reverses a
+2.76e-4 gap.

### 4.1 Where the campaign actually is (MEASURED, decomposed — never a bare composite)

The own-vehicle exact-protocol line, n600 `evaluate.py`, byte-closed, `[macOS-CPU advisory]`:

| axis | value | quantity | share of S |
|---|---:|---|---:|
| seg `100·d_seg` | **0.431179** | d_seg 0.00431179 | 44.7% |
| pose `√(10·d̄)` | **0.292941** | d̄_pose 0.00858145 | 30.4% |
| rate `25·B/37,545,489` | **0.239866** | 360,238 B | 24.9% |
| **S** | **0.9639878179** | residual vs prediction **1.82e-6** | — |

**Arithmetic re-derived from components, not read off a summary field** (operating manual §4):
`100·d_seg + √(10·d̄_pose) + 25·B/37,545,489` on the three measured quantities reproduces the receipt
`0.9639878179` to **1.86e-11**. Both gc13 corners re-derive too (C 0.16183 vs 0.16182; D 0.14418 exact),
and the per-axis debts sum to 0.802162 = S − corner C exactly. No number below is a quoted composite.

Bar `min(0.15, official 0.172141)`. **Gap = 0.791847.** gc13's two bar-feasible corners:

| corner | d_seg → seg S | d_pose → pose S | bytes → rate S | S |
|---|---|---|---|---:|
| **C** (bar-feasible, slack 0.0103) | 6.0e-4 → 0.0600 | 2.33e-5 → 0.01526 | 130,000 → 0.08656 | **0.16182** |
| **D** (the sub-0.15 corner) | 2.97e-4 → 0.0297 | 2.33e-5 → 0.01526 | 149,000 → 0.09921 | **0.14418** |

**Per-axis debt to corner C, and the multiple each axis must move:**

| axis | live | corner C | absolute debt | **required multiple** |
|---|---:|---:|---:|---:|
| **seg** | 0.431179 | 0.0600 | **0.371179** (largest) | **7.19×** |
| **pose** | 0.292941 | 0.01526 | 0.277677 | **19.19×** |
| **rate** | 0.239866 | 0.08656 | 0.153305 | **2.77×** (360,238 → 130,000 B) |
| sum | | | **0.802161** | |

### 4.2 THE DECISIVE ARITHMETIC — no banked composition reaches the bar

This is the single most important number in this memo and it is a subtraction anyone can check:

```
banked pose fallback (gc14 §13)                 0.12689
best byte-closed rate ever built (Knee-B,
  174,578 B = 25·174578/37545489)             + 0.11624
                                              ---------
                                                0.24313   with ZERO seg budget
bar                                             0.17214
                                              ---------
EXCESS                                        + 0.07099   ⇒ EXCLUDED
```

**Consequence.** There is no ordering, no waterfill, and no composition of *currently banked*
components that lands under the bar. Pose's banked fallback alone consumes **74%** of the bar
(gc14 §13), and the best rate archive we have ever byte-closed consumes another 68% of it. **Every
axis must improve multiplicatively; none of them is optional, and no single named action crosses
the bar.** Any roadmap that ranks one axis as "the" path is arithmetically wrong. This is the
formal statement of the standing operator law *NO AXIS PRIORITY near-parity* (07-31) and of ja1's
joint-exchange-rate allocator: rungs fire on realized joint exchange rate, never on axis identity.

### 4.3 The four surviving axes — measured ΔS potential, cost, dependency position

| axis | best MEASURED / DERIVED reach | cost | dependency position | honest label |
|---|---|---|---|---|
| **SEG-CAPACITY** (QA24 from-birth re-burn; burn-2 stack QA83/QA84/QA86) | **≤ −0.098 seg+rate is a LOWER BOUND** (gr1 post-hoc cell_drop50, byte-closed `a6398e44`, n600 realized d_seg 0.004310); a from-birth solve "can only EXCEED" it. Burn-2's own reach is **UNMEASURED**. | ~4–13 h local, **$0 cash**, operator-GO | **PARALLEL** — a CAPACITY pool, does not compete for the byte budget (ja1 §6 Build 3). Invalidates everything downstream ⇒ run it alongside, not after. | **MEANS.** The only lever with capacity-scale reach on the binding axis. **Does NOT cross the bar alone** (0.098 / 0.792 = 12%). |
| **SEG-CONTINUATION** (burn-4 window_04+) | **CLOSED.** r = 0.310 measured; geometric remainder **0.00946 S**; w04 goes **net-POSITIVE (+0.00068)**. | ~2 h/window | — | **DEAD by the Contrarian bound** (P·O 0.0095 ≪ 0.05). gc14 §5.4 STOP rule fires at w04. |
| **RATE** (wr1 #766 Knee-A/B; gr1 granularity; r7 coder) | Knee-B **174,578 B = rate 0.11624** (byte-closed). Knee-A realized gate **REJECTED at INSTANCE scope** (+0.153 net: d_seg +0.165, d_pose +0.185 from stale pose params); ck1 shows recovery parity **0.98×**. gr1 knee **SATURATED** at cell_drop50 (restore +0.047 / drop-more +0.052). | ~1 h re-solve + 30 min photo re-fit per token-base change | **FIRST in the order DAG only for non-token-base moves**; a token-base change invalidates pose + photo + selector (Knee-A law). | **MEANS.** Corner C needs 130,000 B — even Knee-B is **34% over**. Rate is NOT solved. |
| **PHASE-FAITHFULNESS** (#425 · #535 · W1-COH) | **NONE BANKED on any carrier.** #425 raster 10,682 B / STORE 37,158 B — d_seg `OWED_through_R_n600_AB`. #535 2,400 B — Fisher-spectrum `[REQUIRED BEFORE BUILD]` never measured. W1-COH reach **≤ 0.00167** at 12–26 KB (B/err 0.075–0.141 vs water **1.273**). | $0 for the two owed preflights; 10–37 KB of rate if built | Fires **only** on a PIERCE-DOMINANT verdict (gc14 §11); **dominated** under DRAIN. | **MEANS, currently DOMINATED.** Lane holds 43.6% of floor mass (13.1× its corner-C allocation) but no carrier has ever banked a d_seg for it. Queue the two $0 preflights regardless of branch. |
| **TERMINAL POSE** (#366 / su2 / QA43; QA68 expert menu) | Banked fallback **0.12689** = **74% of the bar** ⇒ terminal solve **MANDATORY**. Live v4d d̄ 0.00858145; per-pair spread **784×**, **90% of pose mass in 88 pairs (14.7%)** ⇒ CONTENT-limited, not storage-limited (ja1 §4). | ~1 h re-solve. **us1 F2 `camera_fl=910` is NOT an input — the live coder already uses it** (`xi_pose_coder.py:70`); only F4 polyphase remains open. See the §5 self-correction. | AFTER the seg trunk is conditioned (#383 gate); AFTER any token-base change. | **MEANS, MANDATORY.** 2nd-largest absolute debt (0.2777). QA68 expert-menu targets exactly the 88-pair tail and is **DERIVED-but-UNBUILT**. |
| **RESET-OPERATOR / FRESH-vs-WARM** (gc15 #816; gc14 R1 #815) | gc15 **DERIVED from source**: `mlx.optimizers.Adam` defaults `bias_correction=False`, never overridden at `train_tr1:1543` ⇒ η(1)=3.16, **peak η(12)=6.57**, τ=1000 steps ⇒ each boundary injects **1,212.6 extra sign-steps = 16.17 epochs of free displacement**. Predicts all four gc14 boundary observations without being fitted to any. gc15's own S **UNMEASURED, plausibly 0.011–0.047**. | **$0.** `bias_correction=True` is a **ONE-FIELD falsifier** (sets η ≡ 1 exactly). | Gates the *interpretation* of every seg number this month; must be decided **inside** burn-2's config, which inherits the same defect. | **MEANS, decisive-at-$0.** On the leading derived hypothesis, **the campaign's only measured seg descent this month is an artifact of a missing bias correction.** Also a **standing-law violation**: `v←0` + no bias correction ⇒ a metric-free sign-only step, forbidden by `generic_basis_metric_never_optimal`. |

### 4.4 The re-coherence — what changed, and what the queue must stop pointing at

**gc14 did NOT retire the seg axis. It retired ONE MECHANISM on it.** This distinction is the whole
re-coherence, and conflating the two is the failure mode this audit exists to prevent:

- **Retired:** *continuation windows* as a seg actuator (r = 0.310, w04 net-positive, P·O 0.0095).
- **NOT retired:** the **seg debt itself**, which at **0.371179 S is the largest single absolute debt
  in the campaign** and whose axis alone (0.431179) is **2.5× the entire bar**. Seg cannot be handed
  off; it can only be moved by a different actuator.
- gc14's default branch **B5-C** ("hand the slot to RATE") is correct as a **marginal-efficiency**
  call for the *next window* — cell_drop50's banked −0.098 S is 5.4× anything burn-4 produced — but
  it is **not** a bar-crossing path, and §4.2 proves rate cannot become one. Read B5-C as *"stop
  buying seg with continuation epochs,"* **not** as *"seg is done."*
- The actuator that B5-C's freed slot should reach for is **already built**: burn-2. ja1 ranks the
  QA24 re-burn as the **only** seg-axis descent and explicitly places it **PARALLEL** (capacity pool
  vs byte pools) — so it does not even compete with the rate work B5-C wants.

**The ordering that follows (from ja1 §6's order-of-operations DAG, re-anchored):**

```
PARALLEL TRACK A (capacity, no byte-budget competition)
  burn-2  ──►  QA83 head × QA84 rowband grammar × QA86 config corrections
              × gc15 bias_correction DECIDED EXPLICITLY (either value, but declared)
              ⇒ this ONE run is simultaneously the seg-capacity move,
                 gc15's one-field falsifier, and gc14's R1 control.

SERIAL TRACK B (bytes/pose, cheapest-invalidation first)
  1. ja1 RE-ANCHOR at v4d          ($0)  ← its own declared trigger has fired
  2. terminal pose solve            (~1h) ← consume us1 camera_fl=910 + 2×2 polyphase
  3. QA68 per-pair expert menu      (1–3 KB) ← the 88-pair content tail
  4. token-base / rate moves        LAST  ← invalidates pose + photo + selector
```

**Why ja1's re-anchor is not optional.** ja1 §8 pre-registered its own re-anchor trigger verbatim:
*"it re-anchors when v4d lands (adopt v4d's base + re-run the builder)."* **v4d landed and was
MEASURED** (QA78, S 0.9639878179). ja1's ranked table is therefore **stale at its own declared
trigger**: its ranks 1–4 (QA66, QA72a, QA65, QA70/QA69) were all **consumed by v4d**, leaving only
QA68 (UNBUILT) and QA24 (heavy) live. Since the memo's standing instruction is *"the next charter
should read the top of the committed table BEFORE spawning,"* every charter spawned from here reads
a table whose top four rows are already spent. **This is a live mis-pointing of the queue, it is
$0 to fix, and it is the cheapest single coherence repair available.**

---

## §5 THE RANKED NEXT-3 (each labelled MEANS vs COULD-CROSS-THE-BAR)

**No action below crosses the bar.** §4.2 proves the bar needs simultaneous multiplicative movement
on all three axes; the honest ranking is therefore by *absolute debt × readiness × information*, and
every row is labelled **MEANS**. Saying otherwise would be the means-as-ends violation.

### 1. FIRE BURN-2 AT THE WINDOW_03 BOUNDARY (~18:40Z) — not a window_04
**MEANS.** Largest reach on the binding axis; the only capacity-scale seg actuator; **BUILT + TESTED
+ 4 commits landed + never fired.** Config must **explicitly declare `bias_correction`** (gc15), which
makes this single run simultaneously (a) the seg-capacity move, (b) gc15's one-field falsifier, and
(c) gc14's R1 restart-cadence control — three pre-registered questions for one slot.
- *Reach:* UNMEASURED; QA24's post-hoc lower bound is −0.098 seg+rate and a from-birth solve can only
  exceed it (gr1). **12% of the gap at that bound — does NOT cross.**
- *Cost:* ~4–13 h local, $0 cash, operator-GO (CONTAINMENT).
- *Falsifier:* burn-2's n600 endpoint d_seg fails to beat the warm r1c endpoint 0.004264 ⇒ from-birth
  is dominated on this vehicle (bc1 already measured from-birth ep399 at 0.005169, i.e. **worse** —
  so this falsifier has real teeth and burn-2's new grammar/head is exactly what must overturn it).
- *Blocker to clear first:* gc14 **R2** (add a GT-reference term to `UNDRIV_EROSION`; derive the
  estimator window instead of `n_points=5`) — otherwise any guard fights convergence-to-GT.

**NO-FAKE verification of this recommendation (re-derived, not recognized — operating manual §4).**
A "built" claim is testimony until traced, so I traced it against the LIVE trainer rather than
trusting the b2b memo:
- `renderer-head-mode` / `renderer_head_mode` → **8 occurrences** in
  `experiments/train_tr1_partition_renderer_mlx.py` (the live launch-path trainer, pid 49743's binary).
- `rowband` / `RowBandGrammar` → **20 occurrences** in the same file.
- DSL levers present: `src/tac/witness_dsl/qa84_rowband_grammar_20260731.py` and
  `src/tac/witness_dsl/spec_tr1_burn2_20260731.py`.
⇒ **burn-2 is genuinely wired into the live trainer and held by the DSL** — it is a real, fireable
config, not a phantom build. The orphan is a *slot decision*, not a wiring gap.

**Independent confirmation of gc15's mechanism (my own re-derivation, not a citation).**
`grep -c bias_correction experiments/train_tr1_partition_renderer_mlx.py` → **0**. The trainer never
sets the field anywhere, so `mlx.optimizers.Adam`'s default `bias_correction=False` binds unopposed at
every one of the six `optim.Adam` construction/save sites. This reproduces gc15's central
source-inspection claim from a different direction (absence-of-field rather than call-site reading) and
raises V5's status on the *mechanism* leg from single-source to **two-route corroborated**. It does NOT
raise the *causation* leg — that still needs the A/B, which is exactly what firing burn-2 with the field
declared would deliver.

### 2. RE-ANCHOR ja1's JOINT ALLOCATOR AT THE v4d BASE — $0
**MEANS (apparatus).** The standing pre-charter table every future charter is instructed to consult
is stale at its own pre-registered trigger; its top four rows are spent. Re-running the builder on
the v4d base restores the one instrument that prevents axis-reflex mis-allocation. Cheapest coherence
repair in the campaign.
- *Reach:* 0 S directly; it re-orders everything downstream.
- *Falsifier:* the re-anchored table returns the same ranking ⇒ the staleness was moot (ja1's own
  falsifier pattern, which has fired before for ms3/ms4).

### 3. TERMINAL POSE SOLVE — routed by ja1's CONTENT-limited finding (QA68 over QA65) — ~1 h
**MEANS, MANDATORY.** Pose is the 2nd-largest absolute debt (**0.277677 S**) and its banked fallback
alone eats **74%** of the bar, so gc14 §13's "terminal solve MANDATORY" is arithmetic, not preference.
ja1 §4's DERIVED finding routes the effort: pose is **CONTENT-limited**, not storage-limited (784×
per-pair spread, 90% of mass in 88 pairs), so **QA68's expert menu (UNBUILT) outranks QA65's finer
quanta** — the axis-reflex over-weights precision here.
- *Falsifier:* the per-pair expert menu fails to beat the single-expert baseline on the 88-pair tail
  ⇒ the content-limited diagnosis is wrong and pose is storage-limited after all.

> **⚠ SELF-CORRECTION — I had this recommendation WRONG in draft, and the correction is itself a
> finding (operating manual §6; recorded rather than silently fixed).**
> I first wrote this item as *"consume us1's two UNCONSUMED inputs — `camera_fl = 910` + yuv6 2×2
> polyphase"*, following us1 F2/F4 and gc14 **R6**. I then traced it instead of trusting it:
> **`src/tac/boundary_math/xi_pose_coder.py:70` reads `_NATIVE_FX, _NATIVE_FY = 910.0, 910.0`**, and the
> live v4d receiver builds its `K` from exactly that path (`experiments/inflate_runner_v4d.py:61,147`
> → `pfs1_warp_receiver.intrinsics_native`). The value also appears in `src/tac/camera.py:96`
> (`COMMA_INTRINSICS_NATIVE`), `geodesic_pose.py:21-22`, `calibrated_geometry.py:55-56`,
> `raft_pose.py:84`, `lane_mark_pose_v2.py:164,254`. **The live pose path already uses the true focal
> length.**
> **Consequences, stated plainly:** (a) **gc14 R6's focal-length leg is a NO-OP** — its falsifier ("the
> GN does not improve when the true focal length replaces whatever it currently assumes") is already
> answered, because nothing was assuming anything else. R6 survives only on its **F4 polyphase** leg.
> (b) **us1's "FORGOTTEN" taxonomy has a classification defect**: it means *absent from live pose
> DOCTRINE (the memos)*, not *absent from the CODE*. Those are different failures with different cures
> — doctrine-absence is fixed by a citation, code-absence by a patch — and conflating them produced a
> downstream routable (R6) written against a premise the code already satisfies. **Recommended: us1's
> FORGOTTEN rows gain a `code_state` column (`present` / `absent` / `divergent`) so the distinction
> cannot be lost again.** My own near-miss is the evidence that it does get lost.
> (c) A first grep of `camera_fl` returned **179 files** and looked like overwhelming consumption; every
> one was the substring `camera_float`. The false-positive and the true-positive pointed opposite ways
> and *both* were wrong about the question actually being asked. Re-derive, don't recognize.

**$0 riders that should ride whichever branch fires:** gc14 **R3** (adopt the §5.4 rate-triggered
STOP rule into the window decision record — inputs already exist in every decision JSON) · gc14
**R2** (the two predicate re-calibrations) · the **#535 Fisher-spectrum** measurement (its own
`[REQUIRED BEFORE BUILD]` gate, $0, converts a never-fired design into a priced one) · the **W1-COH
Bayes-floor preflight** owed to Yousfi.

---

## §6 PRIOR-LAW PREDICTION LINES (anti-re-anchor discipline — stated before composing, diffed after)

| law | what it ALREADY predicted | diff vs cn3 |
|---|---|---|
| **velocity-driven orphaning** | fast-moving campaigns strand landed work that nobody re-reads | **CONFIRMED, and at a surface the law had not been applied to: a BUILD.** Every prior instance was a *finding* orphaned. burn-2 is 4 landed commits + 40 passing tests + a named consumer, orphaned by a slot decision. **New corollary: a BUILD with a named consumer and no fire receipt is an orphan with a higher price than an orphaned finding, because its cost is already sunk.** |
| **deferral-scatter → defer-at-source, ONE canonical ledger** | per-item records fail recall; one ledger or nothing | **PARTIALLY HONORED — and the residue is structural.** The QA ledger works (95 rows, defer-at-source respected). But **TWO ledgers exist** (task `#NNN` vs QA), and #815/#816 landed in neither/only-one. The law needs its own corollary: **the defer-at-source rule must name WHICH ledger.** |
| **staleness-is-a-named-confound (freshness at consumption)** | consume fresh; input-hash lineage; fail closed | **NEW INSTANCE, and the strongest kind: a stale ALLOCATOR.** ja1 is the instrument that ranks everything else and it declared its own re-anchor trigger, which then fired unobserved. A stale *measurement* misleads one decision; a stale *allocator* misleads every decision after it. |
| **conditional-validity re-grade (rv1)** | changed preconditions re-open descent-conditioned verdicts | **APPLIES NARROWLY, exactly as gc14 §12 found.** Because the descent is boundary-localized, the flat-amplitude family is NOT re-opened. cn3 adds: the same narrowness applies to the QUEUE — rows deferred "until something descends" are **still deferred**, because nothing descended *within a window*. |
| **magnitude-dismissal discipline / relative significance** | small deltas are dismissible only with the ratio stated | **HONORED and load-bearing.** burn-4's −0.018303 is 2.3% of the gap; gc15's own reach is 0.011–0.047; the largest banked number remains QA24's 0.098. All ratios stated. [magnitude-ok on dismissals below 0.001 S.] |
| **fleet-cap-is-a-cap-not-a-quota** | the respawn reflex is the tree-vortex | **HONORED.** This arm spawned **2** read-only audit arms over non-overlapping surfaces (consumption vs queue) and did the roadmap itself. No third arm was spawned for "more coverage." |
| **non-additive pools** | same-pool levers COMPETE, never sum | **LOAD-BEARING in §4.2.** The banked-composition subtraction is only legitimate because pose-banked and rate-Knee-B are in *different* pools; the memo says so rather than silently summing. Within-pool rows (gr1 knee vs QA07 vs QA08) are explicitly marked SATURATED, not added. |
| **NO AXIS PRIORITY near-parity (op 07-31)** | rungs fire on joint realized exchange rate, never axis identity | **CONFIRMED and sharpened.** §4.2 gives the arithmetic *reason* axis-priority is wrong here: no single axis can cross the bar, so any single-axis roadmap is excluded a priori, not merely sub-optimal. |
| **generic_basis_metric_never_optimal (op 07-29)** | derived-or-raced binding; generic defaults are controls, never optima | **NEW INSTANCE FOUND BY gc15, consumed here:** `v←0` + `bias_correction=False` is a uniform-magnitude, sign-only, **metric-free** step that arrived as a *library default* and was never derived or raced. cn3's roadmap makes declaring it a burn-2 config requirement. |

---

## §7 WHAT I COULD NOT DO / OWED

- **`TaskUpdate` is not available in this environment.** Per the charter's explicit fallback, the
  full disposition table is recorded in §3 of this memo and **nowhere else** — a future agent with
  TaskUpdate must transcribe it into `.omx/state/canonical_task_status.jsonl`. Stated plainly so it
  is not mistaken for a completed ledger mutation.
- **Zero scorer forwards.** Every ΔS in §4 is either MEASURED-elsewhere (cited to its receipt) or
  DERIVED. No number in this memo is a fresh measurement of my own.
- **burn-2's reach is UNMEASURED.** §5 item 1 ranks it first on *readiness × axis-bindingness*, not
  on a measured ΔS. If burn-2's grammar/head does not overturn bc1's from-birth ep399 0.005169 (worse
  than warm 0.004264), the ranking is wrong and rate/pose should take the slot outright.
- **I did not verify burn-2's 40 tests still pass on current HEAD.** The commits are cited from its
  memo (`f28e427dd9 · e8d531e735 · 4bdd72a2f7 · d138df0c00`); a re-run is owed before firing.
- **The 0.9639878179 line is `[macOS-CPU advisory]`**, not a contest row. Its distance to the pointer
  (0.1910828242) is not a gap the campaign has ever closed on this vehicle; §4's gap arithmetic is to
  the BAR on the own-vehicle line, and the pointer remains a separate, borrowed-lineage object.
