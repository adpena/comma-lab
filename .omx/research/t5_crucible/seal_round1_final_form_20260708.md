---
doc_type: t5_crucible_seal_round1_final_form_verdict
role: SEAL ROUND 1 of 3 (counter 0/3) — the FIRST round on the FINAL LAUNCHING FORM (v6.4 draft +
  the LawRef migration §14.5) and the FIRST under requirement U (Opus, THREE co-equal lenses).
  Target = the complete launch stack as it fires: DRAFT_OPTIMAL_STACK_v6 @ v6.4/LawRef state +
  witness_autoconfig.derive_crucible_v6_config (LawRef-consuming) + launch_witness_run.py
  (constants-manifest hookup) + the LawRef mechanism (witness_dsl/lawref*.py + canonical_equations/
  evaluators.py). LENS A recursive-adversarial (bugs) · LENS B deep-math meat hunt · LENS C confound
  hunt (req U, co-equal).
date: 2026-07-08
target: DRAFT_OPTIMAL_STACK_v6_20260708.md @ v6.4 + §14.5 (LawRef migration) + the CODE at HEAD
  (working tree CLEAN — git status empty on all five touched files, so working == HEAD by
  construction).
verdict: CLEAN — 0 BLOCKER + 0 MAJOR + 0 MINOR + 2 nits (a τ_end ladder-tag characterization + the
  meta-confound characterization; NEITHER changes a decision, consumer-read number, or build item).
  Counter advances to 1/3.
axis: all numbers [macOS-CPU/MLX advisory]; pointer contest-CPU 0.19110 UNMOVED — this verdict is MEANS.
review_status: fresh-eyes-reviewed(1, final-form) — this verifier authored none of v1..v6.4, the
  LawRef migration, the probes, the prior verdicts, or the code commits. Every load-bearing claim
  below is [re-executed] (live launcher dry-run; transcribed-source law replicas at full precision;
  resolver positive-control probes; artifact sha; targeted pytest) or [verified-on-disk] against the
  primary artifact.
verdict_scope: no NEW negative asserted; the two nits are INSTANCE-level characterizations.
  # VERDICT_SCOPE_OK:no-new-negatives-only-characterizations
---
`[no-triality]`

STORES CONSULTED: ORCHESTRATION_LEDGER.md (reqs A–U; seal rules P6; the v6.4/LawRef landing folds
L247–332; the T value-ladder, R scoping, U confound lens) · DRAFT_OPTIMAL_STACK_v6_20260708.md (FULL
— §0.1–0.4, §1.0/1.1/1.4a, §2.2f/g/2.5, §3.4, §4c, §5, §7c, §9, §10, §11/12, §13, self-attack, §14/
14.3/14.4/14.5) · seal_round1_v62_verdict_20260708.md (FULL — the regression spec: all 6 findings) ·
lawref_migration_crucible_v6_20260708.md + lawref_constant_compiler_351_20260708.md (FULL) ·
payload_tto_350_20260708.md landing fold (B-DET preflight item + owed n600 confirms — verified §1.0
F-DET row + §10 B-DET row reflect them; both are named pre-GO/headroom-gated, non-blocking) · LIVE
SOURCE this session: src/tac/witness_autoconfig.py (derive_crucible_v6_config L1438–1617, value-
identity guard L1521–1528, _CRUCIBLE_V6_DELTAS L1346–1413) · src/tac/witness_dsl/lawref_builtins.py
(FULL — the 4 CONSUMED + LIBRARY LawRefs) · tools/launch_witness_run.py (crucible route L597–601,
write_constants_manifest L457–482) · experiments/train_levelset_witness_realized_through_R_mlx.py
(_softmax_temp_for_epoch L2341–2386 · _hosc_beta_for_epoch L2318–2338 · _lr_scheduled_for_epoch
L2389–2413 · LR denominator wiring L6099–6106/6622–6626 · argparse defaults L7437/7438/7464/7469/
7474) · the mod32cap CONTROL launch.sh (on-disk) · tau_knee_ptau2_20260708.json (on-disk).
EXECUTED $0: ONE real launcher --dry-run (n600/3000ep, --config crucible_v6 → launch.sh walked
token-by-token + constants_manifest.json) · three schedule-law full-precision replicas (τ/β/LR
transcribed from trainer source) · crossing-arithmetic replica · resolver positive-control probes
(sha / mod48 fail-closed / missing-artifact fallback) · pytest 55/55 (crucible+lawref subset) ·
ruff F (clean). NO launches, NO training; run dirs read-only.

# SEAL ROUND 1 (FINAL FORM v6.4 + LawRef) — CLEAN. Counter 1/3.

## §0 REGRESSION — the v6.2 round's six findings + the v6.3/v6.4 folds — ALL HOLD [re-executed]

| prior finding | fix claimed | this round [re-executed] | verdict |
|---|---|---|---|
| MAJOR-1 wrong-surface V pin | `--curriculum-plateau-windows 5` DROPPED | token ABSENT from launch.sh (grep); V=5 binds B1 spec only | ✓ HELD |
| MAJOR-2(i) β misprint 1.41 / unpinned | `--hosc-beta-end 10.0`; misprint→1.7252 everywhere | emitted `--hosc-beta-end 10.0 --hosc-beta-anneal linear`; β(726)=**3.175725**≈control 3.177 (≤0.1%); anti-targets 1.7252 (un-pinned linear) + 1.4122 (cosine misprint) reproduce + test-pinned | ✓ HELD |
| MAJOR-2(ii) AdamW LR sibling unnamed | v6.4 BUILD `--lr-anneal-epochs 1000 --lr-hold-frac 1.0` | emitted; LR-vs-control **max\|Δ\|=0.0** over [1,726] (bit-identical); deviation shared-den 2.831×/3.410× reproduces (draft 2.83/3.41); control `--epochs 1000` VERIFIED on disk | ✓ HELD |
| MAJOR-3 re-anchor leg unmaterialized | `--curriculum-reanchor-levers` added; chroma→start 300 | emitted `--curriculum-reanchor-levers`; chroma `--seg-chroma-boundary-start-epoch 300` | ✓ HELD |
| MINOR-4 min-stage unpinned | `--curriculum-min-stage-epochs 250` | emitted 250 | ✓ HELD |
| MINOR-5 B19 supersession marks | applied §7c/fold-row 8 | present; no dither flag exists (grep 0) | ✓ HELD |
| MINOR-6 §1.1 base label | sealed/store-nothing base + A/B named | §1.1 amended; base-delta CLOSED (§2.3) | ✓ HELD |

Round-2 BLOCKER-1 (τ materialization) still correct: τ(600)=**0.3100030**, τ(650)=τ(675)=τ(726)=
**0.31 exactly** (held); rebound anti-target den-600 plain-cosine τ(675)=**0.3363482** / τ(726)=
**0.3826296** reproduce; control cross-check τ(650)=**0.3098205** (den 1000/end 0.05, VERIFIED from
the control launch.sh). Pose block + F-DET + `--per-group-grad-clip` + no-dup-flags all present.
Launcher: **106/106 flags exist**, no C13 refusal, mem-preflight **67.61 GiB PASS**.

## §1 LENS A — recursive adversarial (bugs). 0 findings.

- **Dry-run walked token-by-token**: every draft knob-table row materializes; NO invented flag
  (launcher never-invent gate 106/106); NO duplicate long flag (grep + `test_..._no_duplicate_long_flags`).
- **The last shared-denominator sibling is closed.** The v6.2 round's open blast radius was the
  three schedules reading the SHARED `--anneal-epochs` (τ/β/LR). This round confirms the FINAL FORM
  splits them correctly: τ (den 3000, cosine_hold@0.2 → holds 0.31), β (den 3000, linear, end 10.0),
  LR (OWN den via `--lr-anneal-epochs 1000`). The trainer LR wiring L6099–6106 (`lr_anneal_epochs =
  args.lr_anneal_epochs or anneal_epochs`) + `_lr_scheduled_for_epoch` (L2389) match the draft's law
  1:1. No NEW sibling exists (visco/seed/eikonal/EMA/l7 all inert or den-free — re-swept).
- **Byte-identity**: working tree CLEAN on all five files → working == HEAD. The LawRef path is
  proven == literal path by the in-derive value-identity guard (L1521) + `test_crucible_v6_lawref_
  resolved_path_is_byte_identical_to_literal_path`; the emitted launch.sh carries zero LawRef tokens
  (provenance-only). The constants_manifest.json is written beside launch.sh, schema
  `constants_manifest.v1`, 4 constants.
- **Test quality high**: the guard tests pin the anti-targets (2178/0.886/rebound for the schedule,
  1.7252+1.4122 for β, 2.83×/3.41× for LR, drift for the value-identity guard, mod48 for
  conditionality, missing-artifact for fallback). A one-token regression of any pinned schedule/pose/
  base/LawRef value is caught. 55/55 green; ruff F clean on all touched files.

## §2 LENS B — deep-math meat hunt. 0 findings.

- **Crossing arithmetic reproduces unrounded** [re-executed, decimal-30]: pose 0.0173205081; rates
  0.0619861/0.0539559; v3 0.1997336 (NO); central **0.1897336** (m 0.0013664); win9 **0.1817034**
  (m 0.0093966); train bars 0.0010137/0.0010940; ILC 9.9573e-4; leverless-remaining 6.135635e-4/
  6.938661e-4 — every digit matches §0.2/§0.3. The LawRef migration is provenance-only and moves NO
  number (confirmed).
- **Smooth-only asymptote verified — and the draft is RIGHT where a reflex says otherwise.** The
  draft's S_asymptote(smooth-only) = 100·1.5795e-3 + pose + rate = **0.2372616 / 0.2292313** is
  correct: the 1.5795e-3 locked mass is already a DECODED floor, so g_dec is NOT re-added. My first
  recompute added g_dec (→ 0.2477) — the same double-count the r1-v6 CLEAN round caught in a
  charter; the draft's composition is the correct one. No finding (a verification WIN).
- **k_max / TAIL budget** at the final constants: gross floor(2350/387.1)=6, net floor((2350−250)/
  387.1)=5 — reproduce; the `--curriculum-min-stage-epochs 250` pin makes the net-5 arithmetic true.
- **LR deviation math** (the WHY-the-build number): control annealed 1e-3→2.57e-4 (den 1000);
  shared-den-3000 stays ~8.9e-4 (2.831×/3.410× at ep675/726) — reproduces exactly, retained as
  anti-target. Req-I co-resident levers: no new interaction the fold dropped (the τ-hold × forfeit-
  arm seam + cadence antagonism are already registered §2.5; the migration adds no lever).

## §3 LENS C — CONFOUND HUNT (req U, co-equal). 0 findings; 1 load-bearing meta-confound characterized.

### 3.1 Apparatus-validity table — the 4 MANIFESTED constants + the window-law anchors

| consumed constant | instrument | axis/key VERIFIED | positive control | conditionality verdict |
|---|---|---|---|---|
| **τ_end 0.31** | `tau_knee_ptau2_20260708.json` field `launch_tau` | sha256 **9898d8d7…** == manifest record (on-disk match); `launch_tau`=0.31 | `launch_tau_within_band=true` on the INDEPENDENTLY-computed knee band [0.19072,0.54294]; control ep650-best τ=**0.3098205** (VERIFIED from control launch.sh den 1000/end 0.05) | VALID (config_tags schedule:mod32cap == target → PASS). NIT-1: `launch_tau` is a DECISION field read-back, not an independent knee output (the knee emits `tau_star`/`f_target`) → the `measured_anchor` tag is slightly generous, but 0.31≈the measured ep650-best inside the independently-derived band ⇒ value corroborated, changes nothing |
| **β_end 10.0** | literal + linear-replica law | β(726)=**3.175725** ≈ control β(726)=**3.177177** (control end 4.0/den 1000 VERIFIED on disk) | the ≤0.1% slope match IS the control | VALID (derived_at_config; re-derive triggers listed) |
| **LR den 1000** | control launch.sh `--epochs 1000` (VERIFIED on disk) | LR **max\|Δ\|=0.0** over [1,726] vs control | trainer defaults `--lr 1e-3 --lr-end 1e-4 --warmup 1` (VERIFIED L7437/7438/7464) == crucible's explicit `--lr 1e-3 --lr-end 1e-4` == control's (unset→default) | VALID (bit-identical; conditionality = LR trio unchanged) |
| **LR hold 1.0** | literal | control freeze 726 < LR den 1000 → no hold | bit-identity above | VALID |
| ν 0.012653 / settle 237.1 / tail 387.1 / s_fit 1.75/1.30 / adaptive-ε 0.7 | trace_probes / P-CON / draft artifacts | — | — | **LIBRARY LawRefs — NOT emitted by the variant** (verified: absent from launch.sh) ⇒ cannot corrupt the launch config. Their schedule-narrative role was vetted in prior rounds; here they carry zero launch load |

### 3.2 Positive controls on the guard INSTRUMENT itself — all PASS [re-executed]
- (a) **sha256 integrity**: manifest-recorded sha == on-disk artifact sha (9898d8d7…) ✓.
- (b) **config-conditionality fail-closed**: resolving with target `schedule=mod48cap` RAISES
  `ConfigConditionalityViolation` (the P-CT1 protection is LIVE, not inert) ✓.
- (c) **missing-artifact fallback**: a bad `artifact_path` → τ_end resolves to the sealed fallback
  **0.31**, `fallback_used=True`, launch NOT blocked ✓.

### 3.3 Config-conditionality of anchors measured at τ 0.216 (req U(c/d)) — DOCUMENTED, no NEW confound
The control descended τ→0.216; the crucible HOLDS τ=0.31 from ep600. So the forfeit fire-band ep675
+ post-650 d_seg trajectory (measured on the control's descending-τ plant) may not transfer to the
held-τ plant. This is DISCLOSED: the forfeit arm ships advisory + fallback-cap-726 + B-INJ pre-GO
(§2.2f), and §1.4a flags the post-650 erosion as epoch-confounded [INFERRED, Q2-τ/SC-3 adjudicate on
run-1]. The VEHICLE difference (store-nothing base vs the mod32cap control) is the §2.3/§14.3
transfer-risk row; the config_tags guard `schedule` (not `vehicle`), and the vehicle risk is
carried openly. m_q(q90) feeds only the LIBRARY τ*/SC-3 live-law path (run-2), never the launch
constant; the annulus census (locked-mass shares) feeds §0.3 only (no flag). None load-bearing on
the emitted config.

### 3.4 Silent-default scan of the emitted config (req U(d)) — clean
`--verdict-batch 32` (OOM-safe chunked verdict, the CLAUDE.md forbidden-pattern guard) · `--async-
verdict` + `--verdict-pairs 0` (advisory, score-neutral, inherited from store-nothing 205, prior-
round vetted) · `--ema-decay 0.997` (shadow saved). The migration introduces NO new default-on/off
that silently corrupts a run-1 MEASUREMENT (it changes zero emitted flags).

### 3.5 META-CONFOUND (req U(e)) — the value-identity guard: THE LOAD-BEARING LENS-C RESULT
The guard `_val == _lit and type(_val) is type(_lit)` certifies **IDENTITY, not CORRECTNESS**. For
the three literal-pin LawRefs (β / LR-den / LR-hold) the LawRef input IS a literal that DUPLICATES
the sealed `_CRUCIBLE_V6_DELTAS` literal — the guard compares a literal to its own copy (it catches
a copy-paste divergence between `lawref_builtins.py` and the deltas dict, or an artifact that
disagrees for τ_end, but NEVER a value that is wrong-yet-consistent-across-both-sources). What it
would take to defeat it: the SAME wrong value typed into BOTH files (β/LR), or into BOTH the artifact
`launch_tau` AND the sealed literal (τ_end). This is REAL but **fully DISCLOSED** — the migration
claims "VALUE-IDENTITY IS THE LAW / ZERO value changes — provenance-only," never value-correctness;
the correctness defense is the seal's independent recompute, which THIS round performed and PASSED (τ
holds 0.31, β ≤0.1%, LR bit-identical — all re-derived against the on-disk control trace). No
over-claim ⇒ no finding. NIT-2 records the characterization; it changes nothing.

## §4 VERDICT + COUNTER

**CLEAN — 0 BLOCKER + 0 MAJOR + 0 MINOR + 2 nits (both INSTANCE-level characterizations that change
no decision, consumer-read number, or build item: NIT-1 τ_end `measured_anchor` tag reads a
decision field, value corroborated; NIT-2 the value-identity guard certifies identity-not-
correctness, disclosed and correctness independently re-verified this round). Counter 0/3 → 1/3.**
Honesty both directions: the entire v6.2 six-finding blast radius is closed and holds; the v6.4 LR
build reproduces the control LR bit-identically (max\|Δ\|=0.0, the strongest possible cross-check),
so the τ/β/LR JOINT state now matches the trace vehicle at every anchor epoch — the shared-
denominator class that consumed three rounds is fully repaired; the LawRef migration is genuinely
provenance-only (byte-identical, guard fail-closed on mod48 + drift, fallback non-blocking, all
positive controls pass); every crossing/asymptote/schedule/k_max number reproduces at full
precision. The confound lens found the value-identity guard's identity-not-correctness limit but it
is disclosed and the correctness it does not cover was re-derived here and passed. Nothing to fix.

Pointer contest-CPU 0.19110 UNMOVED — this verdict is MEANS.
