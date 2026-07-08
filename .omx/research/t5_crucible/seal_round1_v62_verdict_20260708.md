---
doc_type: t5_crucible_seal_round1_v62_verdict
role: SEAL ROUND 1 of 3 on v6.2 (counter restarted 0/3 after the round-2 NOT-CLEAN; target =
  DRAFT_OPTIMAL_STACK_v6 at its v6.2 state + the BLOCKER-1 materialization code). LENS A recursive
  adversarial (code + draft + fold faithfulness) · LENS B deep-math meat hunt. Priority angles per
  the round charter: (1) the cosine-rebound fix itself + the schedule-sibling-class sweep, (2) the
  autoconfig variant as code, (3) the base-delta residual + full launch.sh walk, (4) crossing
  sensitivity arithmetic, (5) regression on all round-2 fixes + probe-fold faithfulness.
date: 2026-07-08
target: DRAFT_OPTIMAL_STACK_v6_20260708.md @ v6.2 (§14 changelog) + afd86bace
  (witness_autoconfig.derive_crucible_v6_config + launcher route + 7 tests) + fad06d015 (draft fold).
verdict: NOT-CLEAN — 0 BLOCKER + 3 MAJOR + 3 MINOR. Counter stays 0/3.
  The round-2 BLOCKER fix itself is VERIFIED CORRECT at full precision (independent replica
  reproduces BOTH mod32cap measured anchors; dry-run re-executed and reproduces 1:1) — the new
  findings live exactly in the fix's un-swept blast radius: the OTHER consumers of the shared
  --anneal-epochs denominator (hosc-β printed WRONG-SHAPE value; AdamW LR sibling UNNAMED) and the
  other event-trigger constants (V pinned on the WRONG SURFACE per v5's own un-amended row 3(a);
  re-anchor flag omitted). Each changes a config build item or a consumer-read number = the bar.
axis: all numbers [macOS-CPU/MLX advisory]; pointer contest-CPU 0.19110 UNMOVED — this verdict is MEANS.
review_status: fresh-eyes-reviewed(1, v6.2) — this verifier authored none of v1..v6.2, the probes,
  the prior verdicts, or the fix commits; every load-bearing claim below is [re-executed] (live
  launcher dry-run, trainer-source law re-derivation, full-precision python, test run) or
  [verified-by-inspection] against the primary on-disk artifact.
---
`[no-triality]`

STORES CONSULTED: ORCHESTRATION_LEDGER.md (charter/seal rules L44–55 + the round-2 fold L231–254 +
the v6.2 landing fold L1105–1133) · DRAFT_OPTIMAL_STACK_v6_20260708.md (FULL, v6.2 state: §0.1–0.4,
§1.0/1.1/1.3/1.4a, §2.2f/g/2.5, §3.4, §4c, §5, §7c, §9, §10, §11/12, §13, self-attack, §14) ·
seal_round2_v6_verdict_20260708.md (full — regression base; all 6 findings re-traced 1:1) ·
probe_tau2_dither_20260708.md (full — both gate resolutions; fold-faithfulness base) ·
DRAFT_OPTIMAL_STACK_v5_20260707.md (§0.1 row 3(a) V-surface ruling L109; row (b) eps provenance
L159; min-stage L360 — the inherited rows the v6.2 code contradicts) · LIVE SOURCE re-read this
session: src/tac/witness_autoconfig.py (deltas L1314–1343, derive_crucible_v6_config L1367–1463,
trailing block L919–941, _proven_base/_all_levers_base) · tools/launch_witness_run.py (crucible_v6
route L566–570, choices, append-not-clobber L904–909) · src/tac/tests/test_witness_autoconfig.py
(all 7 crucible tests, L614–745) · experiments/train_levelset_witness_realized_through_R_mlx.py
(_softmax_temp_for_epoch L2341–2386 · _hosc_beta_for_epoch L2318–2338 · the Muon freeze block
L6517–6541 · LR schedule L6586–6596 + --lr-schedule default True L7432 · _evt_reanchor_epoch
L2049–2072 + gate L5875–5886 · chroma consumption L3600 · event-knob defaults L7693–7719 ·
seed/visco laws L1389/L1734) · src/tac/witness_dsl/gauge.py (HANDOFF_NUCLEUS chart L1213–1222) ·
src/tac/witness_dsl/curriculum_dsl.py (AnalyticLaneRenderBand L1645). EXECUTED $0: ONE real
launcher --dry-run (n600/3000ep, --config crucible_v6 → launch.sh walked token-by-token) ·
pytest test_witness_autoconfig.py (51/51) · ruff F (clean) · full-precision python (τ/β/LR laws,
crossing sensitivity, k_max, dwell). NO launches, NO training; run dirs read-only.

# SEAL ROUND 1 (v6.2) — NOT-CLEAN. Counter stays 0/3.

## §0 Regression first (round-2 findings + probe folds + v6.1 errata) — ALL PASS

- **All 6 round-2 findings landed 1:1** ✓ — BLOCKER-1: B-CFG row + code + PROOF (I re-executed the
  dry-run myself: `103/103 flags`, NO C13 refusal, `mem-preflight 67.61 GiB PASS`, every claimed
  token present in launch.sh — reproduces the commit's proof exactly). MAJOR-A2: pose block in
  launch.sh (`--w-pose 1.0 --pose-carrier --pose-carrier-residual-mode table
  --pose-carrier-source generated`) + #314 named in §1.0 + `test_crucible_v6_pose_block_pinned` ✓.
  MAJOR-B1: §0.3 coverage table + sensitivity ✓. MINOR-1: k_max ≤6 gross/≤5 net in §2.2g(b) + §2 ✓.
  MINOR-2: §10 F26 row ✓. MINOR-3: C4 knob row §1.0 ✓ (and the dry-run carried
  `--per-group-grad-clip`).
- **Probe folds faithful** ✓ — 0.31-STANDS quoted with the probe's own numbers (knee band
  [0.190724, 0.542937], f_target 0.861663/0.862512, no-crisp-elbow, directional note, live-law
  deferral) at §1.4a + §7c. B19 expunged from every RUN-1 surface (§1.1 removed-with-comment, §0.3,
  §5 dead, §7c resolved, §10 struck, §12 dead row, §9 re-pointed), kill FORMULATION-scoped with the
  reformulation queue + measured priors — except two stale historical-table cells (MINOR-5 below).
- **v6.1 errata intact** ✓ — §0.3 VEHICLE NOTE + 0.2180 alternative floor; 0.4989 printed at both
  cites (§1.4a-cascade row 3, §11 row 16).
- **req-R scopes** ✓ — every v6.2 negative carries its scope tag; the base-delta question is
  stated-not-silently-decided (adjudicated in §2.3 below, as the draft requested).
- **Tests/lint** ✓ — 51/51 test_witness_autoconfig.py, ruff F clean. Test quality: mentally mutated
  — any one-token regression of the pinned schedule (den, shape, hold-frac, τ_end, 726, 300), the
  pose block, or a duplicate flag is caught; the τ-law replica bridge is guarded at the trainer end
  by experiments/tests/test_levelset_theta_star_tier2_levers.py (imports the REAL
  `_softmax_temp_for_epoch`), so replica drift cannot go silently green.

## §1 Angle 1 core — THE COSINE-REBOUND FIX ITSELF: VERIFIED CORRECT [re-executed]

Independent re-derivation from the trainer source (L2370–2386: `prog_t=(ep-1)/max(ae-1,1)`,
cosine UNCLAMPED past the denominator — confirmed; `cosine_hold` returns `end` at
`prog_t ≥ hold_frac` — confirmed) + full-precision replica:

| claim | draft/code value | my recompute | verdict |
|---|---|---|---|
| τ(600) ≈ 0.31 (descent complete) | 0.31 | 0.3100030 | ✓ |
| τ(675) = τ(726) = 0.31 EXACTLY (hold) | 0.31 | 0.31 / 0.31 exact | ✓ |
| plain cosine den-600 rebound τ(675)/τ(726) | 0.3363/0.3826 | 0.3363482/0.3826296 | ✓ |
| round-2 wrong emission τ(675) (den 3000, end 0.05) | 0.886 | 0.8870 | ✓ |
| replica vs mod32cap MEASURED τ(650) | 0.3098 | 0.3098205 | ✓ |
| replica + Muon-freeze law vs mod32cap MEASURED END τ | 0.2157 | **0.2156895** | ✓✓ |

The last row is the strongest cross-check: the freeze-at-`muon_start_epoch` law (trainer
L6517–6527, FEED-fm) + the replica reproduce BOTH measured mod32cap anchors (ep650-best 0.3098 AND
the END/ep1000 τ 0.2157 = frozen τ(726) at den 1000) — the τ materialization is derived, not
guessed, and its two independent anchors both land. **The BLOCKER-1 fix is correct as shipped.**

## §2 LENS B — the schedule-sibling-class sweep (the round's priority hunt)

**The sweep table** (every schedule in the emitted launch.sh, its denominator, its state at the
absolute anchors, vs the mod32cap trace vehicle the anchors were measured on):

| schedule | denominator | crucible@650 / @726-freeze | control@650 / @726 | verdict |
|---|---|---|---|---|
| τ (softmax temp) | --anneal-epochs 3000 × hold 0.2 | 0.3115 / **0.31** | 0.3098 / 0.2157 | **FIXED** (the round-2 blocker; §1) |
| hosc-β (linear, emitted `--hosc-beta-anneal linear`) | SHARED --anneal-epochs 3000 | 1.6492 / **1.7252** | 2.9489 / 3.1772 | **MAJOR-2(i)**: deviates 1.8×; draft prints **1.41 = the COSINE-shape value** — wrong for the emitted linear shape |
| AdamW LR (--lr-schedule default ON, cosine) | SHARED --anneal-epochs (L6594 "denominator = anneal_epochs") | 9.0e-4 / **8.8e-4** | 3.5e-4 / 2.6e-4 | **MAJOR-2(ii)**: deviates 2.6–3.4×, UNNAMED anywhere; the lr→lr-end anneal effectively NEVER happens before the 726 freeze (AdamW phase runs ≈ peak-constant) |
| Muon LR (--muon-lr-final-frac 0.1) | own den = Muon stage length (2274 vs control 274) | — | — | note only: FIN-stage laws already routed to run-1 F3 with fail-safe caps (§2.2g(d)) |
| eikonal step | --eikonal-weight-end UNSET → constant 0.01 | — | — | inert ✓ |
| visco-eps (_visco_eps_for_epoch) | own arg; eps0 default 0.0, no flags emitted | — | — | inert ✓ |
| seed-anneal (seed_compose_weight) | own flag, default 0 = constant | — | — | OK-by-design: seed withdrawal is owned by SeedIslandEased r\*-release, not the anneal ✓ |
| stage-transition rewarmup | absolute 8-ep post-boundary window | — | — | den-free ✓ |
| EMA decay / weight-entropy λ | constants | — | — | ✓ |
| l7 | --l7-start-epoch 3000 (demote-to-epochs) | — | — | 1 trailing epoch, EMA-protected, declared in the docstring ✓ |
| event-trigger constants | windows/min-stage/rel-eps | — | — | windows = **MAJOR-1** (wrong surface); min-stage = **MINOR-4**; rel-eps default 1e-4 = the recalibrated design value ✓ |

### 2.1 [MAJOR-2] The shared-denominator class is only ⅓-repaired: β misprinted, LR unnamed
Three schedules read `--anneal-epochs` (τ L2370, β L2334, LR L6594 — all "review C2" same-den by
design). The v6.2 fix pinned τ via cosine_hold and NAMED the β residual — but (i) **the printed
β(726-freeze) ≈ 1.41 is the value of the COSINE anneal shape; the emitted config is
`--hosc-beta-anneal linear` (inherited _all_levers_base), whose value is 1.7252** [re-executed;
the cited control 3.177 is itself the LINEAR den-1000 value — the draft's own cross-anchor
confirms the emitted shape]. Wrong in THREE consumer-read places: §14 item 1, the
derive_crucible_v6_config docstring (witness_autoconfig.py L1401), the ledger v6.2 fold entry.
(ii) **The AdamW LR sibling is named NOWHERE**: at den 3000 the fire-band LR is 2.6× the trace and
the freeze LR 3.4× — the AdamW phase [1,726] realizes ≈[8.8e-4, 1e-3] (near-constant peak) where
the control realized a genuine 1e-3→2.6e-4 anneal. Net: at the absolute anchors (best~650, fire
[670,700], cap 726) the (τ, β, LR) joint optimizer state matches the trace vehicle **in τ only**
— and the ep650-best/ν(tau)=0.012653/settle-237/fire-band laws were all measured at the control's
JOINT state. The trainer structurally couples the three (no per-schedule denominators), so the
admissible dispositions are: (a) pin the β trajectory to the trace — `--hosc-beta-end 10.0`
(linear, den 3000) reproduces the control's β(ep) on [1,726] within 0.1% (freeze 3.176 ≈ 3.177),
with a named latent-hazard note (if the Muon freeze ever becomes event-movable, β would climb past
4); LR has NO flag-level rephasing (fixing it needs a trainer denominator split — a named build
item) so it takes an explicit risk row + a run-1 sensitivity consumer; or (b) accept both with the
same named risk row. Either way: config/build-item content + consumer-read numbers change ⇒
NOT-CLEAN. **hosc-β adjudication (as charged by the round):** β≈1.73-frozen does NOT re-trip the
annealed-hosc divergence class (that failure is fixed-β=4 saturation at START; a LOW frozen β is
smoothness, not divergence) — but it is an UNMEASURED activation regime on every vehicle we have,
sitting exactly at the epochs whose anchors assume trace physics; per the draft's own
absolute-anchor logic (round-2's assumption-challenge: "the config family SCALES, the trace
anchors are ABSOLUTE") the β trajectory should be pinned to the trace or explicitly risk-rowed —
"carry silently at a misprinted value" is the one indefensible option.

### 2.2 Angle 4 — crossing-sensitivity arithmetic: REPRODUCES; honest read recorded
[re-executed, full precision]: pose 0.0173205081; central rate 0.0619861 / win9 0.0539559; bars
central 1.0136635e-3 / win9 1.0939661e-3 ✓; leverless remainder 6.135635e-4 = **60.53%** /
6.938661e-4 = **63.43%** ✓ exact. k_max: gross 6, net-of-FIN 5 ✓ (and coincidentally 5 at EITHER
min-stage 250 or 150 — see MINOR-4). Consistency with the smooth-reachable 2.0351e-3
decomposition: the win9-survives condition requires the decoded residual 3.6146e-3 to fall to
≈1.198e-3 with the leverless 4.001e-4 permanently retained ⇒ ≈75–78% conversion of ALL addressable
mass (smooth 2.0351e-3 + covered-locked 1.1794e-3) jointly. The ledger's own math calls this
plausible only as the ENGINEERED TAIL — and the draft says so plainly (§0.2 "run-1's central
expectation ≈0.26 does NOT cross"; §9 dual probability 2–6%/8–15%). Honest read: the sensitivity
row is arithmetically sound and its optimism is correctly priced by the draft's own probability
model. No finding.

### 2.3 Angle 3 — the base-delta residual: ADJUDICATED (the draft asked this round to decide)
Walked the full emitted launch.sh against the draft's knob surfaces [re-executed dry-run]. The
control-vehicle deltas are NOT composed (emitted: eikonal 0.01, freq-along 4, n-dir-freqs 2,
lane-paint ON, l7@3000 — matching the draft's stated inheritance posture). **Adjudication: the
sealed/store-nothing base is the RIGHT vehicle** — mod32cap is a council-designed CONTROL whose
deltas deliberately disable the growth machinery (lane-init no-op, growth losses off); composing
`Mod32SegOnlyControlBase` would ship the control's deliberate lobotomy as the launch vehicle,
contradicting the capstone intent. No inherited base token contradicts a v6.2 design row EXCEPT
the §1.1 sketch line `base=Mod32SegOnlyControlBase()` ITSELF, which mislabels what materializes
(MINOR-6: amend the sketch line to the sealed/store-nothing base, keep the one-line
`--dsl-lever Mod32SegOnlyControlBase` A/B for the trace-authority check). The real content of the
authority conflict — the ν/τ/anchor laws were measured on the control VEHICLE as well as the
control SCHEDULE — folds into MAJOR-2's named transfer-risk row (same row, one more axis).

## §3 LENS A — fresh adversarial findings on the materialization

### 3.1 [MAJOR-1] `--curriculum-plateau-windows 5` recalibrates the WRONG SURFACE — v5's own
un-amended row forbids exactly this. v5 §0.1 row 3(a) [verified-by-inspection, L109]: the B1
co-predicate window V has "NO in-trainer flag" (it is a witness_control/B1-spec parameter; CT-1's
invented flag was corrected), and "the sister EXISTING flag `--curriculum-plateau-windows`
(default 4) is the EP_LOSS-plateau window, a DIFFERENT surface — **NOT changed** (silently
recalibrating it would be the per-epoch-normalization bug class)". v6 inherits this row un-amended
and §2.2g(c) re-affirms "B1 spec carries both, no silent recalibration". The v6.2 code emits the
trainer flag = 5, labels it "B1/forfeit-arm co-predicate V=5", and the test pins it. Consequences:
(a) the CE→TAU ep-loss trigger now needs 5 plateau windows (≈ +25 ep later fire, bounded by the
300 cap — a schedule change no v6 law derives); (b) the ACTUAL B1 V=5 is NOT delivered by this
token (it lives in the B-INJ/witness_control build, exactly as v5 said). Fix: drop the flag from
`_CRUCIBLE_V6_DELTAS` + the test row (V=5 stays where v5 put it: the B1 spec/B-INJ) — OR land an
explicit draft amendment owning the ep-loss-window recalibration with its own derivation. Bar:
config build-item content + a pinned-wrong test ⇒ NOT-CLEAN.

### 3.2 [MAJOR-3] The re-anchor leg of the event-triggered design is not materialized. §1.1 ships
`AnalyticLaneRenderBand(start=350, boundary_relative=True)` and the landed HANDOFF_NUCLEUS design
(gauge.py L1213–1222) pairs `--curriculum-event-triggered --curriculum-nucleus-guard` WITH
`--curriculum-reanchor-levers` (persistence-warmup + seed-anneal + analytic-band track the FIRED
boundary; "the fixed 275 stagger is unnecessary — re-anchor replaces it"). crucible_v6 emits the
event trigger + nucleus guard but NOT the re-anchor flag (trainer default False, L7725; gate
L5880), and this omission is NOT in the materialization note's named-exclusion list (FusedRKernel/
AACoverageRender/B16/B17′/GNSpectrumProbe/V-trailing are; re-anchor is not). Under the draft's own
early-fire expectation (settle 3/ν_CE = 150.3 < cap 300), the band engages up to ~100+ ep after
the fired boundary instead of +50, and persistence-warmup/seed frames stay hardcoded —
`boundary_relative=True` is silently false in the emitted run. Sister gap: ChromaBoundarySharpen
`start="tau_fire"` has NO re-anchor path at all (chroma is not in the trainer's re-anchor set,
L2049–2072 docstring enumerates persistence-warmup + seed-anneal + analytic-band only) — the
design token is unrealizable as written; emitted = absolute 300. Fix: add
`--curriculum-reanchor-levers` to the deltas (1 token) + either re-word the chroma row to
start=300-absolute or name the chroma re-anchor as a trainer build item. Bar: config content +
a §1.1 design row ⇒ NOT-CLEAN.

### 3.3 MINORS
- **[MINOR-4] min-stage unpinned:** the draft's dwell law prints "shipped min-stage 250 = 13×
  margin"; the emitted config leaves `--curriculum-min-stage-epochs` at trainer default **150**
  (margin 7.81×, still ≥1; k_max net coincidentally unchanged at 5: floor((2350−150)/387.1)=5).
  Pin 250 (1 token) or re-derive the printed constants at 150.
- **[MINOR-5] B19 expunge completeness:** §0.1 fold-row 8 still ends "DECISION: RUN-1, gated" and
  §7c's P-DZ disposition cell still says "B19/P-DITHER run-1 gated lever" — both need the ◆v6.2
  supersession mark that §0.3/§5/§10/§12 carry (historical-table cells, but §7c is a live consumer
  surface and its own P-DITHER row two lines down contradicts the P-DZ cell).
- **[MINOR-6] §1.1 base label** (from §2.3): amend `base=Mod32SegOnlyControlBase()` to the
  materialized sealed/store-nothing base; close the OPEN base-delta question with the §2.3
  adjudication + keep the A/B lever named.

## §4 VERDICT + COUNTER

**NOT-CLEAN — 0 BLOCKER + 3 MAJOR (wrong-surface V pin · re-anchor/boundary_relative gap ·
shared-denominator siblings: β misprinted ×3 as 1.41 [true linear value 1.7252] + AdamW LR
deviation unnamed) + 3 MINOR (min-stage 150-vs-250 · B19 supersession marks · §1.1 base label).
Counter stays 0/3.** Honesty both directions: the round-2 BLOCKER fix is verified CORRECT and
sharp — the τ materialization reproduces both measured mod32cap anchors independently, the
dry-run proof reproduces exactly, the tests are genuine class guards, all six round-2 findings
landed 1:1, both probe folds are faithful, the v6.1 errata held, and every crossing/sensitivity/
k_max number reproduces at full precision. Every new finding lives in the fix's un-swept blast
radius — the OTHER two consumers of the denominator it pinned, and the OTHER event-trigger
constants adjacent to the one it set. All fixes are cheap (≈3 tokens dropped/added in
`_CRUCIBLE_V6_DELTAS` + one β-pin-or-risk-row decision + test/draft edits); none moves the
crossing arithmetic, the probe verdicts, or any gate disposition — but each changes a launch-config
build item or a consumer-read number, which is the bar. The v6.3 fold should re-run this round's
sweep table against the new tokens (2 minutes: dry-run + the three law replicas).

Pointer contest-CPU 0.19110 UNMOVED — this verdict is MEANS.
