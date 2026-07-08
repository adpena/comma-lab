---
doc_type: t5_crucible_seal_round3_final_form_verdict
role: SEAL ROUND 3 of 3 — THE FINAL CERTIFYING ROUND (counter was 2/3 after round-2 CLEAN). A CLEAN
  round SEALS the launch stack at 3/3. Target = the final launching form: DRAFT_OPTIMAL_STACK_v6 @
  v6.4/§14.5 (LawRef migration) + witness_autoconfig.derive_crucible_v6_config (LawRef-consuming) +
  tools/launch_witness_run.py (constants-manifest) + the LawRef mechanism. Requirement U: THREE
  co-equal lenses. Fresh angles vs r1/r2: STEP-0 B-DET n600 governed run + LENS A certification-diff
  (git delta since r2 + FULL test families, not subsets) + LENS B win9 CONDITIONS chain adversarial
  (incl. the pose-3e-5 provenance) + LENS C the pre-GO checklist AS the confound surface + the
  literal GO-command executability.
date: 2026-07-08
target: DRAFT_OPTIMAL_STACK_v6_20260708.md @ v6.4 + §14.5 + CODE at HEAD (working tree CLEAN; HEAD ==
  the round-2 fold 1fa48fb8b ⇒ NO target-file delta since round 2 ⇒ working == HEAD by construction).
verdict: CLEAN (= SEALED 3/3) — 0 BLOCKER + 0 MAJOR + 0 MINOR + 3 nits (1 fresh r3 + 2 carried), NONE
  changing a decision, consumer-read number, or build item. Step-0 B-DET gate did NOT close (governor
  genuinely REFUSED by 2.8 GiB, NOT bypassed) → carried as pre-GO item #1 per the task's explicit
  fallback; op-class determinism already DEFINITIVE (N=5, #350), so no divergence, no launch-blocker.
axis: all numbers [macOS-CPU/MLX advisory]; pointer contest-CPU 0.19110 UNMOVED — this verdict is MEANS.
review_status: fresh-eyes-reviewed(3, final-form) — this verifier authored none of v1..v6.4, the LawRef
  migration, the probes, or the prior verdicts. Every load-bearing claim is [re-executed] (governed
  B-DET run; live launcher dry-run; full pytest families; unrounded crossing replica; artifact sha;
  source read) or [verified-on-disk].
verdict_scope: no NEW negative asserted; the fresh nit is an INSTANCE-level honesty characterization.
  # VERDICT_SCOPE_OK:no-new-negatives-only-characterizations
---
`[no-triality]`

STORES CONSULTED: seal_round1_final_form_20260708.md + seal_round2_final_form_20260708.md (FULL — the
regression spec + both nits, re-verified; angles they took are NOT re-walked) · DRAFT_OPTIMAL_STACK_v6_
20260708.md (§0.2/0.3 crossing+req-N · §1.0/1.1 MAJOR-A2 pose block · §5 rate · §7c/§10 build+pre-GO ·
self-attack §13) · payload_tto_350_20260708.md (FULL — the B-DET seam: the exact one-command +
governed-admission path + the op-class N=5 determinism result) · ORCHESTRATION_LEDGER req-T ladder
classes · LIVE SOURCE: tools/mlx_gpu_determinism_probe.py (`_composite_trainer_argv` → safe_run
routing L276–333) · tools/launch_witness_run.py (dry-run) · witness_autoconfig.derive_crucible_v6_config
· lawref_builtins.py · git log/status. EXECUTED $0: B-DET n600 composite via GOVERNED safe_run
admission (2 attempts, REFUSED — recorded) · launcher dry-run (106/106, ADMIT, manifest) · pytest
target-subset 54/54 + full broad families (3 orthogonal pre-existing failures isolated) · ruff F clean
(5 files) · crossing arithmetic re-executed unrounded · constants_manifest field-by-field · GO-command
executability dry-run. NO training launches; run dirs read-only; scratch dry-run dirs removed.

# SEAL ROUND 3 (FINAL FORM v6.4 + LawRef) — CLEAN = SEALED 3/3.

## STEP 0 — B-DET n600 COMPOSITE DETERMINISM — GATE NOT CLOSED (governor genuine refuse); CARRIED pre-GO #1

Ran `tools/mlx_gpu_determinism_probe.py --composite --comp-pairs 600 --comp-fused-r` via the SANCTIONED
governed path (`_composite_trainer_argv` wraps the real trainer in `tools/safe_run.py --projected-gib
55.0 --rss-mb 74000`; NOT a raw bypass). Physical free was **76 GiB** (stable), but the P0 SUM-over-RAM
crash guard **REFUSED on 2 attempts**: *"projected system-used 102.2 GiB EXCEEDS adaptive ceiling 99.5 /
99.4 GiB by 2.7 / 2.8 GiB (current used 22.2 + active-growth reserve 25.0 + new 55.0)"* (ledger
`experiments/results/b_det_composite_350/composite_hashes.jsonl`). The 25.0 GiB active-growth reserve is
a STABLE governor policy (persisted across both attempts, ~identical) — NOT the decaying #350-STAGE-2
psutil transient. **I did NOT bypass** (lowering `--projected-gib` would game a gate whose 55 GiB is the
honest #350-measured monolithic self-orient footprint; killing the 2.5 GiB dashboard would not close a
2.8 GiB gap). Per the task's explicit fallback: **record the number, carry the gate as pre-GO item #1.**

Why this is CLEAN-compatible, not NOT-CLEAN: (1) NOT-CLEAN is reserved for "divergence with the failing
op named" — **no divergence occurred**; (2) the determinism ANSWER is already **DEFINITIVE at op-class**
— #350 measured GPU+fused-R+self-orient **BIT-IDENTICAL N=5** (0 unique hashes), and #348 ruled out
kernel-selection-by-size (32×2^18 bit-identical) ⇒ determinism is per-op scale-invariant; the n600 cell
is a true-scale belt-and-suspenders confirm, not the primary evidence; (3) B-DET is documented
non-blocking either way (§10/§7c L618: "launch NOT blocked either way" — proofs fall back to CPU-locked
if it ever diverged). SC-21 row is thus **partially satisfied** (op-class definitive; n600 true-scale
confirm owed). B-DET N-at-scale: **not attained (governor-refused; 0 procs hashed)** — carried.

## §1 LENS A — the CERTIFICATION DIFF (fresh: git delta + FULL test families + manifest field-by-field). 0 findings.

- **Cert diff — zero target-file delta since round 2 [verified-on-disk].** `git status` empty; `git log`
  shows HEAD == the round-2 fold `1fa48fb8b` (the only commits since r2's verdict are the r2 verdict + its
  fold, both docs-only) ⇒ NO change to any of the five target files ⇒ every round-2 claim carries forward
  un-reopened. Working == HEAD by construction.
- **FULL test families (not subsets).** Broad net (`-k "crucible or lawref or autoconfig or witness_dsl
  or launch or witness_control or constants_manifest or payload_tto or stage3 or exact_ab"`) = **489
  passed, 3 failed**. The 3 (`test_hinerv_hard_region_miner::…flips_launch_gate_to_l5`,
  `test_launch_lightning_alpha_geo0…requires_active_dispatch_claim`,
  `test_nerv_long_training_campaign_admission…accepts_snerv`) are **PRE-EXISTING + ORTHOGONAL**: none
  imports any target file (grep 0), all last touched early June 2026 (a month before the crucible work),
  in HiNeRV/SNeRV/Lightning campaign infra (one carries a hardcoded `now_utc="2026-06-02"` — a
  time/dispatch-claim bug, not a crucible regression). The **isolated target subset = 54/54 green**
  (matches r2's 54/54). Ruff F clean on all 5 files. The 3 failures change no decision/number/build-item
  for the launch stack ⇒ out-of-scope characterization, not a finding.
- **Manifest field-by-field vs req-T ladder [re-executed].** Dry-run wrote `constants_manifest.v1`, 4
  constants: τ_end 0.31 `ladder_class=measured_anchor` (sha **9898d8d7…**, fallback_used=false, config_tags
  schedule=mod32cap) · β_end 10.0 / lr_anneal_epochs 1000 / lr_hold_frac 1.0 all `derived_at_config`,
  fallback false. `measured_anchor` + `derived_at_config` ARE the req-T ladder classes (ledger L258/304).
  Emitted launch.sh tokens (τ/β/LR/anchors/pose MAJOR-A2 `--w-pose 1.0 --pose-carrier … generated`/chroma
  reanchor 300/`--verdict-batch 32`/`--fused-r-kernel`/`--mlx-device gpu`/`--per-group-grad-clip`)
  identical to the round-2 verified set. 106/106 flags, no C13, mem-preflight 67.61 GiB PASS.

## §2 LENS B — the launch-case arithmetic, adversarial (fresh: the win9 CONDITIONS chain incl. pose provenance). 1 fresh nit.

- **All four crossing cases reproduce EXACTLY, unrounded** [re-executed]: v3 (0.0011,3e-5,central) =
  **0.1997336** NO/+0.0086336 · v6 central (0.0010,3e-5,central) = **0.1897336** YES/margin 0.0013664 ·
  **v6+win9 (0.0010,3e-5,win9) = 0.1817034 YES/margin 0.0093966** · win9-at-old (0.0011,…) = 0.1917034
  NO/−0.0006034. Inputs: g_dec +1.0427e-4; pose √(10·3e-5)=0.0173205081; central rate 93,092 B →
  0.0619861, win9 rate 81,032 B → 0.0539559. Train bars central ≤0.0010137 / win9 ≤0.0010940; ILC chain
  9.9573e-4. Req-N asymptote reproduces: S_asymptote(smooth-only)=100·1.5795e-3+pose+rate=**0.2372616
  central / 0.2292313 win9** — and g_dec is CORRECTLY NOT re-added (the locked mass is a DECODED floor;
  the r1 double-count trap avoided). Family asymptote S≈0.165 band [0.154,0.181] retained with lower
  edge explicitly CONDITIONAL on locked-mass coverage (measured NEGATIVE prior for dither via P-DITHER).
- **The win9 CONDITIONS chain — each verified measured / gated / disclosed [re-walked]:** the d_seg
  train bar (≤0.0010940) is run-1-measured (central expectation ≈0.26 does NOT cross — the whole §0.2 is
  the OPTIMISTIC engineered tail, stated plainly); the THREE named binding constraints (anneal-completion/
  endpoint recovery · lane composed-band efficacy · locked-mass coverage) are all disclosed + the
  leverless-25.3% quarter is measured (=0.0400 S, 4.26× the win9 margin if unrecovered); win9 81,032 B /
  central 93,092 B rate provenance is §5 inherited byte-close; disposition already "conditional — this
  measures the condition and tightens it." No silent d_seg assumption.
- **FRESH NIT (r3): the pose 3e-5 is a FOURTH, least-evidenced crossing condition not named in §0.2's
  "THREE binding constraints" and booked WITHOUT the borrowed-ancestor caveat the campaign's own memory
  demands.** §0.2 books pose at 3e-5 in all four rows and says "NO probe fold moves rate/pose/g_dec,"
  treating pose like a fixed input alongside rate/g_dec — but per MEMORY L68/L69 witness d_pose is
  **OPEN/UNMEASURED** (warp 3.7–10.3; the ancestor 3.4e-5 "don't cite SOLVED, HELD until byte-close;
  0.018-equiv pose = BORROWED ancestor"). A pose miss to 3e-4 (√(10·3e-4)=0.0547722) would blow the win9
  margin by 4×. WHY IT IS A NIT, NOT A MINOR: (a) the pose MOVER is correctly pinned + run-1-ACTIVE
  (`--w-pose 1.0 --pose-carrier-source generated`, `real_keyframe` excluded, regression-tested) ⇒ the
  condition IS run-1-measured, satisfying the task's disjunction ("measured / gated / disclosed as
  run-1-measured"); (b) round-2 §3c disclosed run-1 EVSI is pose-dominated (~0.044 of ~0.05) ⇒ pose is
  acknowledged the big run-1 unknown; (c) it changes NO config/number/build-item. It is a §0.2 NARRATIVE
  completeness gap → a one-line fold caveat ("pose→~3e-5 = a run-1-measured, currently ancestor-borrowed
  FOURTH condition, MEMORY L68/L69"), not launch-blocking.

## §3 LENS C — the pre-GO checklist AS the confound surface (fresh: two-gate asymmetry + GO executability). 0 contradiction findings.

- **The safe_run-vs-launcher admission ASYMMETRY [re-executed] — a real characterization the operator
  must see.** The launcher's OWN system-admission **ADMITS** the 67.61 GiB crucible run right now
  (projected 90.8 ≤ ceiling 98.4, headroom 7.6) — while safe_run **REFUSES** the SMALLER 55 GiB B-DET
  probe, because the two gates differ: safe_run adds a +25 GiB active-growth reserve; the launcher's
  calibrated admission does not. Consequence: B-DET is on a STRICTER gate than the launch itself — the
  r1-inferred coherence "box that admits crucible admits B-DET" is FALSE; B-DET can refuse on the very
  box where crucible launches. This does not block anything (B-DET is non-blocking + optional true-scale
  confirm), but the runbook must present B-DET honestly as "run opportunistically when the box has
  ~+28 GiB extra headroom over the launch; op-class determinism already definitive." Characterization,
  changes no build item.
- **No pre-GO item contradicts another.** B-DET (fallback: CPU-locked proof paths) · B-INJ (fallback:
  slope-arm + hard cap 726, the operative TAU→FIN transition regardless) · F26 (fallback: constant τ_end
  0.31 + checkpoint-granularity m_q) — three independent fallbacks, none coupled; cap 726 is the operative
  transition whether or not B-INJ fires. Coherent.
- **The operator's single GO command is literally executable [re-executed].** The runbook referenced
  `launch_witness_run.py --config crucible_v6` but never wrote a complete paste-able line (argparse REQUIRES
  `--gt-cache` — a bare `--config crucible_v6 --num-pairs …` errors). The P7 declaration below supplies the
  verified line; I dry-ran it EXACTLY (106/106 flags, system-admission ADMIT, constants_manifest written).
- **Silent-default scan** clean (unchanged from r1/r2): `--spike-guard-mode` unset ⇒ trainer default
  `rollback` (NOT the median-freeze deadlock, L5 memory guarded); liveness `accepted_frac`/`weights_stepped`
  stamped per verdict; `--verdict-batch 32` (OOM-safe chunked); `--async-verdict --verdict-pairs 0`
  (advisory, score-neutral). NIT-2 (identity-not-correctness of the value-identity guard) STANDS as a
  disclosed characterization — correctness re-derived AGAIN this round (§2 crossing + §1 manifest).

## §4 VERDICT + COUNTER

**CLEAN = SEALED 3/3.** 0 BLOCKER + 0 MAJOR + 0 MINOR + 3 nits (r3-fresh: §0.2 pose-3e-5 fourth-condition
caveat; r2-carried: config-docstring V=5 co-predicate conflation + advisory-cadence line; r1/r2-carried:
value-identity guard certifies identity-not-correctness). None changes a decision, consumer-read number,
or build item. Step-0 B-DET n600 gate did NOT close (P0 governor genuinely refused by 2.8 GiB, twice, NOT
bypassed) → carried as pre-GO item #1 per the task's explicit fallback; op-class determinism is already
definitive (N=5 bit-identical, #350) so there is no divergence and no launch-blocker. Honesty both
directions: the cert-diff is zero-delta and the target subset is 54/54 green; every crossing/asymptote
number reproduces at full precision and req-N is correct (g_dec not double-counted); the pre-GO items are
coherent + all carry fallbacks; the GO command is executable. The fresh find is a pose-honesty §0.2
caveat the campaign's own memory demands — foldable, not launch-blocking.

## §5 — THE SEAL DECLARATION (P7 handoff, operator-facing)

**CERTIFIED (3 fresh-eyes Opus rounds, all CLEAN):** the final launching form — DRAFT_OPTIMAL_STACK_v6 @
v6.4/§14.5 + `derive_crucible_v6_config` + `launch_witness_run.py --config crucible_v6` + the LawRef
mechanism — is INTERNALLY COHERENT and BUILD-COMPLETE: 106/106 flags exist, no C13 refusal, mem-preflight
67.61 GiB PASS + launcher system-admission ADMIT, 4 LawRef constants materialize into launch.sh
byte-identically to their sealed literals (guard fail-closed on drift/mod48; fallback non-blocking),
target tests 54/54 green, ruff F clean; the τ/β/LR JOINT schedule reproduces the mod32cap control at every
anchor epoch (τ holds 0.31 from ep600; β ≤0.1%; LR bit-identical) and the anneal-complete ep600 < forfeit
fire ep675 < Muon cap 726 timeline COHERES; the pose block is pinned + run-1-active (store-nothing ξ
carrier, real_keyframe excluded); every crossing/asymptote number reproduces unrounded.

**EXPLICITLY NOT COVERED by this seal:** (1) IDENTITY, NOT CORRECTNESS — the value-identity guard proves
the LawRef path == the literal path; value-correctness rests on this seal's independent recompute (done,
passed), not the guard. (2) Whether run-1 ACTUALLY crosses 0.19110 — §0.2 is the OPTIMISTIC engineered
target; run-1's central expectation ≈0.26 does NOT cross; crossing is conditional on the named d_seg
constraints AND on witness d_pose reaching ~3e-5 (OPEN/UNMEASURED — booked at the ancestor 3.4e-5;
run-1-measured via the active carrier). (3) UNBUILT-DISCLOSED items below.

**PRE-GO CHECKLIST** (none blocks the GO; each carries a fallback):

| # | item | status | command / action | est |
|---|---|---|---|---|
| 1 | B-DET n600 composite determinism | **OWED** (governor-refused now by 2.8 GiB; op-class N=5 already definitive) | `.venv/bin/python tools/mlx_gpu_determinism_probe.py --composite --comp-pairs 600 --comp-fused-r --n 3` — run when box has ~+28 GiB headroom over the launch; **non-blocking** (proofs fall back CPU-locked) | ~10–15 min |
| 2 | B-INJ forfeit injection test | **OWED / unbuilt** (~20 LOC) | synthetic injection through witness_control forfeit wiring (fires-when-should + silent-when-shouldn't); **non-blocking** (slope-arm + hard cap 726 carry) | ~20 min |
| 3 | F26 SC-3-ext live-m_q route | **OWED / unbuilt** (~15 LOC; run-2 capability) | wrap `flip_margin_quantiles` in the trainer verdict block; **non-blocking** (checkpoint-granularity + constant-0.31 fail-safe carry) | ~15 min |
| 4 | docstring tighten (r2 NIT-1) | **OWED / doc** | `derive_crucible_v6_config` docstring L1468–9: separate the live EP_LOSS event-trigger from the manual d_seg V=5 advisory + name the advisory cadence; **non-blocking** | ~2 min |
| 5 | §0.2 pose caveat (r3 NIT) | **OWED / doc** | add inline: pose→~3e-5 is a run-1-measured, currently ancestor-borrowed FOURTH crossing condition (MEMORY L68/L69); **non-blocking** | ~2 min |
| — | NIT-2 identity-not-correctness | **DONE / disclosed** (correctness re-derived r1/r2/r3) | no action | 0 |

**THE ONE-COMMAND GO** (verified executable this round — dry-ran 106/106 + ADMIT + manifest; drop `--dry-run` to fire):
```
.venv/bin/python tools/launch_witness_run.py \
  --config crucible_v6 \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 --epochs 3000 \
  --out-dir experiments/results/levelset_n600_crucible_v6_run1_$(date -u +%Y%m%dT%H%M%SZ)
```
The launcher AUTO-STARTS the score-neutral shadow observer + dashboard; crash-resume re-runs the FROZEN
argv (zero LawRef re-resolution — bit-faithful). B-DET (item 1) is the standing launch-preflight; when it
admits, it runs first. Determinism is already op-class definitive, so a GO now (item 1 carried) is sound.

---
FINAL (≤16 lines):
- B-DET n600: GATE NOT CLOSED — P0 governor genuinely REFUSED (projected 102.2 > ceiling 99.4 by 2.8 GiB;
  active-growth reserve 25.0; 2 attempts identical), NOT bypassed → carried pre-GO #1. Op-class
  determinism ALREADY definitive (N=5 BIT-IDENTICAL, #350); no divergence ⇒ not a launch-blocker.
- LENS A (cert diff): 0 findings — zero target delta since r2 (HEAD==r2 fold); target subset 54/54 green;
  3 broad-net failures pre-existing + orthogonal (no target import); manifest field-by-field correct.
- LENS B (arithmetic): 1 fresh NIT — §0.2 books pose at borrowed-ancestor 3e-5 as an un-named FOURTH
  condition sans caveat (pose OPEN on witness, MEMORY L68/L69); mover pinned+run-1-active ⇒ nit, not
  MINOR. All 4 crossing cases + req-N asymptote reproduce EXACTLY.
- LENS C (pre-GO surface): 0 contradiction findings — items coherent, all non-blocking; safe_run stricter
  than launcher (B-DET refuses where crucible ADMITs — characterized); GO command executable (dry-ran).
- **SEALED 3/3.** 0 BLOCKER/MAJOR/MINOR + 3 nits (1 fresh + 2 carried). Pre-GO checklist: 5 owed items,
  NONE blocks GO (each has a fallback: CPU-locked proofs / cap 726 / constant-0.31 / doc). One-command GO
  supplied + verified. Pointer contest-CPU 0.19110 UNMOVED — this verdict is MEANS.
