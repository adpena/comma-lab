# ddm_qd1 — backlog drain, 2026-08-03

**Pointer UNMOVED.** Live best remains `ddm_cx1` **S = 0.8264972** (seg 0.4311790 · pose 0.1597320 ·
rate 0.2355862 @ 353,808 B) — recomputed from components, not read off a composite: `100·d_seg` and
`25·B/37,545,489` reproduce the quoted seg and rate exactly, and the pose term implies
d_pose = 0.002551431. Gap to the 0.172141 bar = **0.6543562**. This unit was apparatus and triage; it
did not move the exact score and does not claim to.

**Three fixes landed, all $0, all mutation-verified.** Fourteen rows drained with receipts. The single
highest-value catch is not a fix at all: **FIRE-ORDER-0 is dominated** (§B, #826).

---

## A. What landed

| commit | row | what |
|---|---|---|
| `47502f687c` | #907 | `ST_GRID` cross-copy drift detector + de-self-certified `test_ddm_cx1_container_compose` |
| `e5d18538d5` | #856 | the one real import-time GPU allocator closed (`bench_lane_band_cache`) |
| `ffbaa63960` | #899 residual | an UNVERIFIED retirement could drain real debt out of every queue |

### The finding inside #907 worth carrying forward

The row said "hand-duplicated ×5 with one DIVERGED copy." Re-derived at source: **that does not
reproduce.** All vendored copies are byte-identical; the two 10-element ladders are not drift (both
sites score `s_t=0` separately as an explicit null before sweeping the positive grid — verified in
both); the copies are deliberately NOT a refactor target because `pfs1_warp_receiver` documents
itself as needing *"NO tac dependency"* to be vendored whole into the shipping decode path.

What was missing was a DETECTOR — and **my own first detector was blind at the highest-value site.**
`tools/pfs1_recompose_warp_base_and_eval.py` holds the receiver inside an `INFLATE_RUNNER` string
literal that it writes out as `inflate_runner.py`, so **the copy that actually ships lives inside a
string**. A plain AST walk sees a string, not an assignment, and reported "8 sites, all clean" while
unable to see it. Recursing into embedded source took the scan 8 → 9. 4/4 mutations now detected;
vacuity floor included so a shrunken scope FAILS instead of reporting green over nothing.

Note the convergence: the `#856` agent independently found the same genus — **11 of 17 MLX sites bind
`mx` via `pytest.importorskip`, invisible to an `ast.Import` walk (65% miss)**. Two detectors, two
blind spots, same cause: *the detector's AST shape did not match how the code expresses the thing.*

---

## B. #826 — FIRE-ORDER-0 is DOMINATED. Do not fire it as a score candidate.

The ledger row reads: *"gr1_cell_drop50_archive.zip (359,221 B) byte-closed at seg_plus_rate
0.6702284 vs ref 0.7685479 (**−0.0983195**), never through exact eval."*

MEASURED against the live best rather than against its own reference:

| | bytes | rate | seg | **seg+rate** |
|---|---:|---:|---:|---:|
| `gr1_cell_drop50` | 359,221 | 0.2391905 | 0.4310379 | **0.6702284** |
| `cx1` (live best) | 353,808 | 0.2355862 | 0.4311790 | **0.6667652** |
| **gr1 − cx1** | **+5,413** | **+0.0036043** | −0.0001411 | **+0.0034632 (WORSE)** |

The −0.0983195 is real **against its own v4d-era reference** (0.7685479 = the S≈0.9640 v4d base).
Against the live vehicle the candidate is **dominated**: its seg is marginally better (−0.0001411) but
its archive is **5,413 B larger**, and the rate penalty is 25× the seg gain. Firing it as a score
candidate regresses seg+rate by +0.0035.

**Attacking my own comparison.** The soft leg is the seg term: gr1's d_seg (0.004310379) and cx1's
(0.00431179) come from different receipts, and if they were measured on different surfaces the seg
delta is not strictly comparable. **The conclusion does not depend on it.** The rate leg is a byte
count — the most reliable quantity available — and the rate penalty alone (**+0.0036043**) exceeds the
entire seg gain (0.0001411) by **25×**. gr1 would remain dominated even if its seg advantage were
several times larger than measured. The verdict is byte-driven, and therefore robust.

**Fairness to the source (§1 — compose with standing context, do not override it).**
`current_focus.md:50-56` is *more careful than the ledger row*: it already states "full S ~0.96, so it
does **not** move the pointer. Its value is the first genuine exact row for our own vehicle + the
never-done CALIBRATION of our advisory n600 protocol against the real evaluator." **That calibration
rationale survives intact and is a legitimate reason to fire.** What does not survive is the score
rationale, and the ledger row is the caveat-stripped transit of the careful claim — it carries the
−0.0983 headline without "vs which baseline" and without "does not move the pointer." *A number
stripped of its caveat becomes a lie by transit* (§5.2). Consistent with #827's own note that ep854
already dominates gr1's seg base by −0.035996 S.

**DERIVED and actionable:** for `cell_drop50`'s seg mechanism to be net-positive against cx1, its byte
cost must fall from 5,413 B to **≤ 212 B** (0.0001411 · 37,545,489 / 25) — a **25.5× reduction**. That
is the real question the row poses, and nobody has asked it.

**Disposition:** SCORE rationale SUPERSEDED by cx1; CALIBRATION rationale LIVE and owner-retained
(MAIN / burn owner). Fire it *only* labelled as protocol calibration, never as a gap-closer.

---

## C. The ledger split — measured, and worse than "two stores"

`m89` says the harness TaskList and `.omx/state/canonical_task_status.jsonl` are different stores.
**Re-measured today:** the repo ledger holds **415 rows / 148 unique `task_id`s / 62 open**
(pending 43 · blocked 11 · in_progress 8). Numeric ids run 41…871 — **the entire `#874–#911` band the
dispatch cited is ABSENT from the repo store.** Every prompt-named row in this unit had to be located
by CONTENT.

The split is not symmetric and that is the part to remember: **arms see only the repo store**, so a
bare id from the harness sends an arm hunting something that does not exist. Citing content works;
citing ids does not. Structural bridge remains OWED.

---

## D. Full disposition table

Nothing is silently skipped. `RUN` = done this unit. `STALE` = precondition moved.
`SUPERSEDED` = a later measurement dominates it. `BLOCKED` = named, measured blocker + fire-condition.

### D.1 Harness-named rows (located by content; absent from the repo store)

| row | disposition | receipt |
|---|---|---|
| **#899** fail-open P0 gate, read path | **STALE (main claim) + RUN (residual)** | Read path fixed `124a35cae4` (ddm_ri1, 2026-08-02) — `_validate_required_component` extracted so write+read share code; a 3,400×-worse candidate now ranks **3, not 0**; negative control holds; live store 22/22 verified. Residual found by attacking that fix and LANDED `ffbaa63960`. |
| **#904** declared-on-never-read, N-hop | **DIAGNOSIS HALF WRONG — re-scope before building** | gd5's failed F1 detector was **already repo-wide (3,251 modules)**; its measured cause was *"the measured-better-successor relation has no representation in code"* — not scope. The scope diagnosis holds only for the *flag-never-read* class, where `_trainer_consumers` parses ONE file (443 flags, **0 fireable**) and accepts a rebind (`_x = float(getattr(args,"x",2.0))`) as proof of consumption — 141 of 456 dests have only that shape. **Two classes conflated in one row.** |
| **#885** `git log \| wc -l` = 50 vs 13,742 | **RUN — reproduced, mechanism isolated, scope corrected** | See §E. Real, live, and NOT what a repo-side gate would fix. |
| **#856** 12 import-time MLX leakers | **RUN (partial) — framing INVERTED** | 17 sites not 12; **16 are deliberate CPU-pins**, whose hazard is the mirror (importing one pins the whole process to CPU silently). Exactly 1 real GPU allocator, closed `e5d18538d5`. Guard OWED (see §F). |
| **#858** receiver admits ABSENT `token_codec` | **STALE / CLOSED** | `_validate_selector` enforces strict key-set equality (`:302`); `:317` explicitly REFUSES an explicit `null` as *"a second spelling of the legacy framing"*; encode (`:523`) and decode (`:595`) read it symmetrically; `test_legacy_token_codec_stays_absent_and_byte_identical` pins it. Absent==legacy is canonical, single-spelled, tested. The `subagent_contract.py:411` comment that reads like a live bug is HISTORICAL PROVENANCE of the discovery. |
| **#907** `ST_GRID` ×5, one diverged | **RUN — does not reproduce as stated; real hole was different** | §A. LANDED `47502f687c`. |
| **#878** unpersisted NEXT-IF-RESUMED | **SUPERSEDED** | `09fca46f37` (ddm_rs2, today) landed the additive `findings` field + `read --findings`; `next_action` was already persisted at `:164`. Dogfooded by this unit's own checkpoints. |
| **#877** unnamed consumers of 2-decimal `Final score` | **CONFIRMED LIVE — owner needed** | The dispatch's own evidence stands: `evaluate.py` printed `0.83` for BOTH pj2 (0.8308905) and cx1 (0.8264972) — it cannot resolve a 0.0044 move. Correctness hazard, not hygiene. NOT run here (needs a consumer sweep + a decision on the print format, which touches the eval path — deliberately not touched while `ddm_pz1` may claim the scorer slot). |
| **#875** subset defaults under-sample a verdict | **CONFIRMED LIVE, second anchor found** | `m88` (bp2: prefix said −0.122 WIN, n600 said +0.152 LOSS) plus `gd5`'s 36-pair realized gate being a different population from the n600 base (+40.5% Movable). Cure is one line and already known: *report the subset's mean of the governing quantity against the population's; ratio≠1 ⇒ different population.* OWED as a gate. |
| **#874** caps that cannot report why they stopped | **SAME GENUS AS #885/`m50` — fold, do not build separately** | A cap that stops silently and a truncation that reports completeness are one class: *the instrument cannot emit its own limit.* Recommend merging #874 into the #885 law rather than a separate detector. |
| **#883 / #911** serializer/commit integrity | **NOT RUN — genuinely deferred** | Blocker: both are about a repair path that committed a non-empty index and a commit that absorbed 215 unauthored lines. Auditing that safely requires reproducing a serializer failure mode against the live lock, which risks the shared working tree while sibling arms (`ddm_op2`, `ddm_pz1`, `ddm_cu1`) are active. **Fire-condition:** run when no sibling arm holds uncommitted edits (check `.omx/state/subagent_progress.jsonl`). |

### D.2 Repo-ledger open rows (62) — by cluster

| cluster | n | disposition | receipt / fire-condition |
|---|---:|---|---|
| June-18 witness/levelset rows (`boundary_flip_sidecar`, `ego_hood_per_frame_mask`, `final_fine_tune_converged`, `fp_shrink_qat_rate_lever`, `lane_poly_geometric_spatial_prior`, `pose_low_rank_radial_zoom_codec`) | 6 | **SUPERSEDED — vehicle pivot** | Own-vehicle line is now TR1/ix2 (v4d→pw1→ms8→dc1_fold→cx1, `m06`/`m08`); these target the witness/levelset ancestor. NOT killed (CLAUDE.md forbids premature KILL). **Fire-condition:** a TR1-native analogue is designed. `fp_shrink_qat` claims −0.022…−0.029 S on RATE — re-derive on TR1 before believing it transfers (`m07`: ancestor numbers are hypotheses here). |
| JRD cluster (`jrd_n600_tensor_prior`, `jrd_pose_decoupling_r1`, `jrd_training_time_entropy`, `jrd_v9_cgauge`, `jrd_witness_rate_instrument`) + `v9_jrd_coeff_prefix_probe` + `task503_recursive_fractal` | 7 | **SUPERSEDED — V9 vehicle** | All V9/witness-payload scoped; `task503` already carries `V9_INTEGRATION_BLOCKED_OWNER`. **Fire-condition:** V9 payload work resumes as a live line. |
| `deferral_ledger::D41–D53` | 16 | **MIXED — see D.3** | This IS the canonical queue ledger `m36` asks for; correctly parked with owners. One is STALE (below). |
| C1/C4-mod19 checkpoint custody (`AUTH-C4-MOD19`, `C1-WITNESS-CLEAN-STAGE-EMA`, `C4-MOD19-RATE-BYTECLOSE`) | 3 | **BLOCKED — named, measured** | Chain blocks on `C4_BLOCKED_CHECKPOINT_CUSTODY` / no eligible clean C1 witness checkpoint. Witness-vehicle scoped ⇒ also superseded. **Fire-condition:** an eligible clean stage-EMA checkpoint exists AND the witness line is live again. |
| `pact-g111-*` (3) | 3 | **BLOCKED — correctly chained** | Two block on `pact-g111-complete-trainable-state-resume`, which is `in_progress` under codex/root. Resumability is P0 per CLAUDE.md, so this chain is legitimately open. **Not mine — owner-retained.** |
| `ddm_ra1::task_*` (213/227/408/573/611) | 5 | **OWNER = MAIN, un-actioned** | Reframing/binding tasks. Per `m45` these must exit OWNED: they are MAIN's and stay MAIN's; I did not silently adopt or drop them. **Fire-condition:** MAIN's next roadmap pass. |
| Live 8xx cluster (807, 809, 815, 819, 820, 821, 822, 824, 825, 826, 828, 871) | 12 | **see D.4** | The only cluster on the current vehicle. |
| Singletons (`494`, `578`, `575-m1-c2`, `sfess_cached_replay`, `costate_organ_duty_queue`, `phase_residual_carrier_store_half_359`, `einstein_kolmogorov_crux::PDW1_PALETTE`, `schmidt_icml2026_optstep`, `canonical_task_status_ledger_superseded`) | 9 | mixed | `578` blocked on `D2_ZERO_BYTE_SEMANTIC_CELLS_TO_RGB_ADMISSION_FALSE`; `575-m1-c2` blocked on a **memory-admission rc=4 refusal** — that is the governor working, and a REFUSE is information, not an obstacle (CLAUDE.md §D). `sfess` blocked on a managed-sandbox serializer failure. `canonical_task_status_ledger_superseded` is **CONFIRMED + EXTENDED by §C**. |

### D.3 `deferral_ledger::D41–D53` — the one that drains

**`D52a` "Median-freeze convergence-confound cleanup" — STALE.** The median-freeze/spike-guard
deadlock was fixed and gated: `check_no_spike_guard_defaults_to_deadlock_mode` (#397) and its
generalized sister `check_reject_filter_updates_reference_from_accepted_only_has_rearm` (#398) are
both named in CLAUDE.md's confound-immune-system section as landed STRICT gates, and `L4`/`L5` record
the 18-confound hunt as closed with the `ep_loss:0.0` alarm live. **Recommend: mark completed.**

The rest split cleanly and none is READY ∧ high-EV (so none violates the anti-deferral rule):
- **Vehicle-agnostic compute** (D43 sparse-adjoint, D47 costate reuse, D48a/b/c, D50 ANE/CoreML,
  D51 megakernel): survive the pivot but are **MEANS, not exact-row movers** — CLAUDE.md builds
  infrastructure only in service of an imminent exact row. Genuinely deferred; fire-condition = a
  named exact row they unblock.
- **V9-scoped** (D42 K32/64/128 student, D46 SPS, D49 "Current-V9 optimal micro-batch"): SUPERSEDED
  with the V9 cluster.
- **D41** already carries `CONSOLIDATION_OWNER_DRAINED_NO_PROMOTED_IMPLEMENTATION`; **D53** carries
  `missing_exact_canonical_task_or_source_for_transient_495` — both are named measured blockers, i.e.
  correctly deferred by the rule's own test.
- **D44** (converged margin-saliency/taper), **D45** (AdamW/MLX semantics), **D52b** (cured HOSC),
  **D52c** (FreSh): these touch optimizer/activation and overlap `ddm_op2`'s live scope
  (optimizer-state + EMA basis). **Deliberately not touched — collision avoidance.**

### D.4 Live 8xx cluster

| row | disposition |
|---|---|
| **#826** | **SUPERSEDED as score / LIVE as calibration** — §B. The most consequential finding of this unit. |
| **#828** | **OWNERLESS on the current vehicle — highest unclaimed value.** `rehearse_ddm_tr1_runtime.py::_mlx_reference` never sets `_quant_engaged=True`, so past-knee checkpoints compare an UNQUANTIZED reference to a QUANTIZED receiver and emit **false `BLOCKED_DEPLOYMENT`**. `current_focus.md:41` states the principle exactly: *"a gate emitting false BLOCKED is the mirror of one emitting false clean — arguably worse, because it stops real work."* Same genus as this unit's other four instrument catches. **Not run: it is a TR1-runtime file and `ddm_cu1` holds receiver custody.** Fire-condition: `ddm_cu1` reports, or MAIN reassigns. |
| **#807** burn-4 COMPOSE+SEAL+FIRE | in_progress under `ddm_b4s`; owner-retained. Burn is DONE per `m06` (ep399 d_seg 0.0038892 FLAT) — likely closable by its owner. |
| **#871** | in_progress under `ddm_bs2`, already carrying a full measured result (byte-closed 360,323→360,339 B, +16). Owner-retained. |
| **#815 / #824 / #820** | Reset-operator race + its carve-out + the 6-site wiring debt it depends on. **#824 is explicitly STANDALONE + UNBLOCKED (~2h, $0)** and does NOT require #820 — by the anti-deferral rule this is the readiest real experiment in the ledger. **Not run here: it is a training run and overlaps `ddm_op2` (optimizer state).** Fire-condition: `ddm_op2` clears, then fire #824 first. |
| **#819 / #825 / #821** | Build-completeness / registry / hollow-gates apparatus under `ddm_sb2`+`ddm_rg5`. **This unit paid down part of #821's genus** (a hollow gate) via the #899 residual. Owner-retained. |
| **#809 / #822** | cg1 per-class guard re-calibration; lane-guard sign disagreement (`realized_lane_s_units` vs `net_betti0_realized_lane_delta` are different quantities). #822 is a **units confound** — same family as `m66` (ΔS without its denominator). Owner-retained (`cn3`/`lg1` successor). |

---

## E. #885 — reproduced, and the scope is the finding

**MEASURED.** `rtk git log --oneline | wc -l` returns **50** against a true **13,855**
(`git rev-list --count HEAD`) — a **277× undercount**, with **exit code 0 and 0 bytes of stderr**. The
50th line is a real commit; there is no ellipsis, no banner, nothing to notice.

Mechanism isolated: **rtk applies a silent default `-n 50`.** With an explicit `-n` it passes through
(`-n 100` → 100, `-n 20000` → 13,855), and `rtk git rev-list --count` is correct.

**Blast radius, and why a preflight gate would have been the wrong fix.** On a true 200-commit range:

```
rtk git log --oneline BASE..HEAD  | wc -l  ->  50    (150 commits silently dropped)
rtk git log --format=%H BASE..HEAD| wc -l  ->  50
rtk git diff --stat BASE..HEAD             ->  CORRECT (489 files, full range)
```

So `git log` and `git diff` **disagree over the same range and nothing says so** — and the arm-harvest
law `m74` ("harvest FULL `BASE..HEAD`") is silently violated for any arm producing >50 commits.

But repo code is **immune**: Python `subprocess` returns the true 13,855 for **both** the list-arg and
`shell=True` forms. rtk is a Claude Code **Bash-tool hook**, so it never reaches `subprocess`. A raw
grep suggests "47 of 68 `git log` callsites lack `-n`" — that count is **misleading**: most are
docstrings/comments, and every real callsite is immune anyway.

**Therefore the class lives in AGENT BEHAVIOUR, not in code**, and the durable fix is a law, not a
gate: *when counting or enumerating commits from the Bash tool, pass an explicit `-n`, or use
`git rev-list --count`.* Building a preflight gate against 47 immune callsites would have been a
point-fix on the wrong surface (§8.3) — I nearly did it.

---

## F. Owed, named, not claimed

1. **#856 two-landing guard.** Non-trivial: **11 of 17 sites bind `mx` via
   `mx = pytest.importorskip("mlx.core")`**, which is not an `ast.Import` node — a guard walking
   `ast.Import`/`ast.ImportFrom` misses **65%**. Catalog #184 is the right AST primitive but scopes to
   exactly ONE file (`src/tac/preflight.py`) and checks heavy *imports*, not device *calls*.
2. **The CPU-pin hazard** (15 sites): importing any of them silently pins the whole process to CPU.
   That is the *mirror* of #856's stated concern and has no row. Worth one.
3. **The grade-5 surface is itself built-never-fired.** `built_elsewhere_unwired()` is live-EMPTY and
   has **0 production consumers over a 63,014-file sweep** (only the definer and its tests). A P0
   queue nobody reads.
4. **`not_even_designed` unverified-retirement** — closed here, but the sibling test had only ever
   planted a VALID retirement. Worth checking the same "positive-case-only test" shape elsewhere.
5. **UNMEASURED:** whether `mx.metal.is_available()` initializes the Metal backend. If it does, the
   import-time leaker count is materially higher (5 substrate files + 5 tools call it at module scope).

---

## G. Denominators (an empty scope is VACUOUS, never a PASS)

| sweep | denominator | result |
|---|---:|---|
| repo task ledger | 415 rows / 148 ids | 62 open; `#874–#911` band ABSENT |
| `ST_GRID` ladders (src, tools, experiments) | 9 found | 8 checked, 1 intentionally skipped; 4/4 mutations detected |
| MLX import-time scan | 10,348 `.py` | 17 sites; 1 real GPU allocator |
| #899 consumer sweep | 63,014 `.py` | 2 files (definer + tests); **0 production** |
| `corpus_query` stores | research 7,418 · memory 2,059 · dag 915 · tasks 415 · equations 869 · council 292 · docs 96 | index ≈76% — every negative here is scoped to that |
| git history | 13,855 commits | rtk path reports 50 |

---

## NEXT-IF-RESUMED

1. **Do not fire #826 as a score candidate** — dominated by cx1 by +0.0034632 on seg+rate. Fire it
   *only* labelled as advisory-protocol calibration. If someone wants the seg mechanism, the question
   is the **25.5× byte reduction** (5,413 B → ≤212 B), which nobody has asked.
2. **#828 needs an owner** (currently `unassigned`) — a gate emitting false `BLOCKED_DEPLOYMENT` on the
   current vehicle. Gated behind `ddm_cu1`'s receiver custody.
3. **#824 is the readiest real experiment** in the ledger (standalone, unblocked, ~2h, $0). Gated
   behind `ddm_op2`.
4. **Re-scope #904 into two rows** before anyone builds a third detector — the successor-relation class
   and the flag-never-read class have different measured diagnoses.
5. **Mark `deferral_ledger::D52a` completed** (median-freeze confound fixed + gated by #397/#398).
6. **Fold #874 into the #885 law** rather than building a separate cap detector — one class.
7. **#883/#911** when no sibling arm holds uncommitted edits.
