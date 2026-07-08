---
doc_type: t5_crucible_seal_round2_v6_verdict
role: SEAL ROUND 2 of 3 (fresh pass; round 1 on v6 was CLEAN, counter was 1/3) — LENS A recursive
  adversarial verify + LENS B deep-math meat hunt, deliberately on the angles round 1 did NOT take
  (build-list executability · end-to-end schedule simulation · req-P random re-audit · locked-mass
  per-lever coverage · pose+rate legs · LAUNCH-PATH REALITY measured via the real launcher dry-run ·
  req-R substance on the v6.1 errata).
date: 2026-07-08
target: DRAFT_OPTIMAL_STACK_v6_20260708.md carrying the v6.1 errata (census-vehicle note + 0.4989
  width label — BOTH verified landed; regression section below).
verdict: NOT-CLEAN — 1 BLOCKER + 2 MAJOR + 3 MINOR. Counter RESETS to 0/3 on v6.
  The BLOCKER is a launch-path materialization gap MEASURED on the real launcher (dry-run REFUSAL +
  wrong-schedule emission), not prose taste; the fix is small (~30–60 LOC named config/DSL variant +
  a §10 row + a §0.3 coverage table) but it changes build items and consumer-read claims, which is
  exactly the NOT-CLEAN bar.
axis: all numbers [macOS-CPU/MLX advisory]; pointer contest-CPU 0.19110 UNMOVED — this verdict is MEANS.
review_status: fresh-eyes-reviewed(2) — this verifier authored none of v1..v6, the probes, the
  round-1 verdicts, or the hardening sweep; every load-bearing claim below is [re-executed] (live
  launcher dry-runs, census JSON recompute, live argparse/source greps, full-precision python) or
  [verified-by-inspection] against the primary on-disk artifact.
---

STORES CONSULTED: ORCHESTRATION_LEDGER.md (reqs A–S read in full at §93–145 + fold entries; pose /
launch-route greps over the whole file) · DRAFT_OPTIMAL_STACK_v6_20260708.md (full, both pages) ·
seal_round1_v6_verdict_20260708.md (full — regression base; angles deliberately NOT repeated) ·
DRAFT_OPTIMAL_STACK_v5/v4/v3/v2/v1 (schedule §2, ledger §4c 19 rows, rate §5/§5.1, pose §6 chain
v1→v6, build lists §10) · hardening_sweep_20260708.md (launcher atomic-write fix e28ff371e; item-4
v5.1 supersession) · probe_tau2_dither_20260708.md (header — sibling IN-PROGRESS; its untracked
files LEFT untouched per sister-coherence) · position_S6_pose_byteclose_20260707.md (pose flag
provenance) · CONTEXT_COMPENDIUM_20260707.md (#314 DRIFT-D2 row) · LIVE SOURCE re-read this
session: experiments/train_levelset_witness_realized_through_R_mlx.py (pose argparse L7438/L8322–
8350; --anneal-epochs default None L7287; --softmax-temp-end default 0.05 L7394) ·
src/tac/witness_autoconfig.py (pose_carrier_source default real_keyframe L518; store-nothing
variant L1118–1143; sealed pins) · tools/launch_witness_run.py (gate map; --config choices;
duplicate-flag refusal L358; --dry-run) · src/tac/witness_annulus_metrics.py
(flip_margin_quantiles exists; zero trainer callers — grep) · EXECUTED $0 this session: TWO real
launcher --dry-run invocations at n600/3000-ep scale (see BLOCKER-1) · census recompute from
experiments/results/t5_probe_waveB_20260708/pdz_deadzone_census.json (per-class-pair locked-mass
decomposition) · full-precision python (τ-anneal simulation, k_max, byte sums, 0.2180 check).
NOT consulted: durable-state files (stale per sweep); the sibling's in-flight P-TAU2/P-DITHER
work products beyond the memo header (live, untracked — not mine to grade). $0 only: dry-run
preflights + reading + arithmetic; NO launch, NO training, run dirs read-only.

# SEAL ROUND 2 (v6) — NOT-CLEAN. Counter resets to 0/3.

## §0 Regression checks first (round-1 minors + hardening sweep)

- **v6.1 MINOR-1 landed** ✓ — §0.3 decomposition row now carries the VEHICLE NOTE (θ* per-stage-
  attribution run, 2026-06-30, not mod32cap) + the fraction-transfer alternative floor **0.2180**.
  [re-executed]: 0.38366×0.0036146 = 1.38679e-3 → 100×that + 0.0173205 + 0.0619861 = 0.2179856 →
  "0.2180" ✓, still > 0.19110 ✓, bar-vs-locked 1.39× > 1 ✓ — every disposition stands as claimed.
- **v6.1 MINOR-2 landed** ✓ — 0.4989 (= 0.31·ln5 = 0.49893) now printed at BOTH cites (§1.4a-cascade
  row 3 and §11 row 16), old 0.4986 correctly attributed to 0.3098·ln5.
- **Hardening sweep vs v6's launch surface** ✓ no stale claim — v6 §10 B-DET "riding launch
  preflight" is consistent with the launcher architecture post-e28ff371e (atomic launch.sh write);
  v6 makes no claim about the launch.sh write path that the inode fix invalidates.
- nit-3/nit-4 from round 1 were explicitly bound to the P7 editorial fold, not owed here ✓.

## §1 LENS A — recursive adversarial verify (fresh angles)

### 1.1 [BLOCKER-1] The launch route does NOT materialize the v6 schedule — MEASURED, not guessed
(angles 1+6: build-list executability × launch-path reality)

v6 never names the artifact that turns the draft into the launcher invocation. The launcher
(`tools/launch_witness_run.py`) takes `--config {proven_base, all_levers, sealed_205,
store_nothing_205, fresh_seeded}` + `--dsl-lever` (lever appends) + `--extra-trainer-flags`.
None of the five named configs is the v6 stack, and BOTH remaining routes fail, measured by
executing the real launcher in `--dry-run` (CPU-only, no spawn) at the real scale
(gt_n600.npz, `--num-pairs 600 --epochs 3000`, config `store_nothing_205`):

1. **Extras route REFUSED (launch-gate refusal, verbatim):** passing the v6 τ-endpoint via
   `--extra-trainer-flags "--softmax-temp-end 0.31"` → `ERROR: REFUSING to launch — the emitted
   argv would contain DUPLICATE long-flag(s) ['--softmax-temp-end'] (argparse last-wins silently
   shifts schedules; confound C13)`. The sealed family already pins the flag (at 0.05, not 0.31).
   Per the charter's own rule ("any gate that would REFUSE the v6 config as-specified = BLOCKER").
2. **Named-config route silently materializes a WRONG schedule:** the successful dry-run's emitted
   launch.sh contains `--muon-start-epoch 2178` (the family SCALES 0.726×epochs) and **no
   `--anneal-epochs` token at all** — trainer default None ⇒ anneal denominator = `--epochs` = 3000
   (source L7287–7289). Simulated [re-executed]: τ(675) = **0.886**, τ(726) = **0.869**, anneal
   completes ~ep3000. Under v6's own laws the promoted forfeit arm's anneal-complete PRECONDITION
   is false throughout the fire band [670,700]; the cap-726 fail-safe would force TAU→FIN at
   τ ≈ 0.87 — a categorically different run from the draft (which assumes anneal 600 · best ~650 ·
   fire ~675 · Muon ≤ 726 as ABSOLUTE epochs).

So the v6 launch config requires a NEW named autoconfig variant / DSL-compiled config (crucible-v6:
pin `--anneal-epochs 600`, `--softmax-temp-end 0.31`, absolute Muon/cap epochs, `--fused-r-kernel`,
the B1/forfeit-arm advisory wiring, the pose block of BLOCKER-relevant MAJOR-A2) — ~30–60 LOC in
`src/tac/witness_autoconfig.py` following the existing `store_nothing_variant` pattern + one
launcher `choices` entry. §10's F-DET row says "**0 (config)**" LOC — under-scoped: there is no
existing config surface where the v6 values can land. **Bar test: adds a build item + corrects a
build-list LOC/route claim ⇒ NOT-CLEAN by definition.** (What passed on the same dry-runs, for the
record: flag validation 85/85 + `--fused-r-kernel` accepted; memory preflight at v6 scale PASSES —
projected peak **67.61 GiB** vs ceiling 108.8 GiB @ safe-frac 0.85; the system-admission REFUSE I
observed was state-dependent (a concurrent governed job on the box), not a config defect; C4
auto-injected `--per-group-grad-clip` → MINOR-3.)

### 1.2 [MAJOR-A2] The pose leg is unpinned at the draft's config surface; #314 never named
(angle 5, pose)

The crossing books the pose leg at √(10·3e-5) = 0.0173205 through the store-nothing ξ carrier
(pose ON, Track 1, v3 §6). But: (i) **zero pose flags appear anywhere in v3–v6** — grep
`--pose|--w-pose` over all four files = 0 hits; the §1.0 tables claim "every flag" verified while
the pose block is absent; the pin `--pose-carrier-source generated` exists only in v1's knob table,
inherited "verbatim" five documents deep (v2 §6 → v3 §6 → v4/v5/v6 silence). (ii) The live trainer
default is `--pose-carrier-source real_keyframe` (L8327) and `--w-pose 0.0` (L7438) — real_keyframe
is the mover v1 explicitly EXCLUDED ("wrong mover"; L69), and w_pose 0 is pose-blind. (iii) **#314
DRIFT-D2 — the OPEN, SCORE-RELEVANT pose-carrier-source inheritance bug (compendium POSE face) — is
named nowhere in v6 nor in the entire ORCHESTRATION_LEDGER** (grep = 0 hits), despite being a drift
bug on exactly this flag. Mitigation verified live: `store_nothing_205` emits the correct block
(`--w-pose 1.0 --pose-carrier --pose-carrier-source generated` — read from the emitted launch.sh),
so the correct route EXISTS in-tree; the defect is that the launch-candidate document does not bind
to it. Fix: pin the pose block + the #314 guard in §1.0/§1.1 and in BLOCKER-1's new config (~5
lines). Bar test: changes the launch-config build item's CONTENT + closes a named score-relevant
drift class ⇒ seal-relevant. Rate leg, by contrast, SURVIVED the same audit (§3.1).

### 1.3 Schedule coherence end-to-end (angle 2) — one MINOR, rest survives

Simulated 0→3000 with the v6 laws (assuming BLOCKER-1's config fix): CE exit event (settle 3/ν_CE
= 150.3) → TAU with anneal-den 600 → best ~650-class → forfeit arm armed post-600, fires at slope
zero-crossing (any positive s* fires once slope ≤ 0, so s* = 6.8971e-6 sitting a hair below the
quoted [6.9e-6, 1.42e-5] band is immaterial — the invariance comes from the zero-crossing, not the
band edges) → cap 726 > band [670,700] ✓ → FIN (min-stage 250; dwell law ln(1.275)/0.012653 = 19.2
ep, 250/19.2 = 13× ✓) → TAIL_k floors 387.1. **[MINOR-1]** k_max = floor(2350/387.1) = 6 books ZERO
epochs to FIN between fire (~675) and TAIL_1: with FIN ≥ min-stage 250, k_max ≤ floor(2100/387.1)
= **5**. The "turnpike 3–7 survives" statement still holds, TAIL is τ*-limited not budget-limited,
crossing books no TAIL gain — no decision changes; one-clause fix ("k_max ≤ 6 gross / ≤ 5 net of
FIN dwell"). cap_fin/settle-237 coherence ✓ (fire band precedes cap; settle 237 < the ~425-ep TAU).
Verdict cadence stays 25 (B-CT3 unbuilt) ✓; ckpt-every 25 emitted by the config family ✓.

### 1.4 Build-list executability walk (angle 1) — remainder buildable; one MINOR

B-DET (~15, surface = launch preflight, pass/fail semantics in §7c) ✓ · B-INJ (~20, surface =
witness_control wiring, fires/silent scenarios named) ✓ · B17′ (~10 on B17, fitted bars + per-class
lane column; backtest ledger = the SC-17 format already emitted) ✓ · B16 (~25, margin-saliency
surface + DSL Lever + activation-ledger row, gated) ✓ · B19 (~15–25, decode-side; injection point
pinned by the sibling's probe memo to the packet inflate.py `_R` uint8 site — and the P-DITHER
instrument ALREADY EXISTS in-tree as untracked sibling work: tools/witness_dither_decode_ab.py +
tac.witness_control.decode_dither; favorable, left untouched) ✓ · TAIL_k = B14 (~40, v4 row, DSL
TailCycles factory named) ✓ · c(τ) rows = declared-constant + SC-18/alarm instrumentation, no
hidden build ✓ · **[MINOR-2] SC-3-ext live-m_q promotion has NO named generation route/build item:**
"live m_q(t) EMITTED PER VERDICT CADENCE" — but `flip_margin_quantiles` has ZERO trainer callers
(grep; only offline tools call it), and §10's F18–F25 row covers SC-14..19/20/21, not SC-3-ext.
Consumers (τ_end live-law promotion, P-TAU2, TAIL τ*_k live form) are covered at CHECKPOINT
granularity by the committed offline instrument (witness_tau_mq_confirm — the sibling's in-flight
P-TAU2 uses exactly that), so nothing blocks; fix = one §10 row naming the route (in-trainer ~15
LOC wrap OR per-cadence instrument re-run, like SC-20's pattern). Bar: adds a build-list row —
MINOR because the fail-safes (constant 0.31; τ_{k−1}/2) and the offline instrument carry run-1.

**[MINOR-3]** The launcher's C4 emit-side injection `--per-group-grad-clip` (score-affecting,
landed f1dd0cc2f, injected on BOTH my dry-runs) appears nowhere in v6's knob→control-law surface —
the real launch argv will carry a knob the draft never enumerates, against v6's own epistemic
contract ("every knob carries a CONTROL LAW class + tag"). Deliberate, landed, tested — no decision
changes; one knob-table row fix (class (a) constant, V-S, with the `--no-` opt-out named).

### 1.5 Requirement-P random re-audit (angle 3) — PASSES

Five rows traced consumer→surface: SC-2 (TAIL stop rule = B14 law; campaign ILC §8) ✓ · SC-11
(costate SENSE store — #247 controller, tools/costate_digest.py live) ✓ · SC-14 (σ_meas → threshold
SE denominators; F18 ~10 LOC named) ✓ · SC-20 (committed Q1 correlator instrument; consumers §1.3
weight-of-choice + B16 adjudication + L76 ledger) ✓ · SC-21 (B-DET GO gate + proof routing) ✓.
No write-only row found in the sample; the 21-row dedupe (19+SC-20+SC-21) verified ✓.

### 1.6 Requirement-R substance on v6.1's negatives (angle 7) — PASSES

The errata introduced no new negatives; the VEHICLE NOTE is scoped at the narrowest supported
level (instance/vehicle + transfer-convention alternative computed, both conventions > 0.19110).
Spot-re-check of the scope ladder round 1 certified: P-CON kill stays FORMULATION (raw τ·ln5 form;
fitted bar live) · P-MP autopsy correctly splits selection-mechanism (dead) from capacity (passes)
· Q1 S_R withdrawal stays scoped to the point-predictor role. No over-scoped kill found.

## §2 LENS B — deep-math meat hunt (fresh angles)

### 2.1 [MAJOR-B1] Locked-mass per-lever coverage is computable TODAY — and 25.3% of it has NO
large-amplitude lever (angle 4)

v6 §0.2/§0.3 name locked-mass coverage as the third binding crossing constraint, then defer its
quantification to a run-1 SC-16 overlap computation, while claiming "the large-amplitude levers act
at multi-quantum amplitude on EXACTLY the census's concentration regions (far-lane rows / hood /
horizon)". The per-class-pair decomposition is computable NOW from the on-disk census artifact
($0, [re-executed] this round from pdz_deadzone_census.json, HR form, fractions of the 1.5795e-3
locked mass):

| lever family | class-pairs | locked share | locked d_seg |
|---|---|---:|---:|
| lane band + islands + comb | Lane↔Road + Lane↔* | **42.7%** | 6.749e-4 |
| #139 hood clamp | Road↔MyCar (+MyCar↔*) | **19.3%** | 3.042e-4 |
| islands w_movable | Movable↔Road/Undrivable | **12.9%** | 2.032e-4 |
| **NO named large-amplitude lever** | **Road↔Undrivable (horizon/shadow)** | **25.3%** | **4.001e-4** |

The horizon/shadow quarter = **0.0400 S ≈ 22.5× the crossing margin** and is attacked ONLY by the
GATED B19 dither ("plausibly a small fraction") — plus, unlisted in §0.3's lever enumeration,
ChromaBoundarySharpen's g_I-raising mechanism (the deadzone predicate is HR·g_I·m/|∇m| < quantum,
so raising local edge contrast is a locked-mass lever the draft never credits or masks). So the
"EXACTLY the concentration regions" sentence overstates horizon coverage, and the third binding
constraint carries a now-quantifiable hole. Bar test: changes a consumer-read claim (§0.3/§9
coverage wording + the conditional [0.154, 0.181] lower edge tightens), adds a §0.3 table + a
duty-queue/solve-inventory row (an Undrivable-boundary lever question ranks ABOVE several queued
reformulations at 22.5× margin) ⇒ seal-relevant. Crossing DISPOSITION unchanged (the tail was
already conditional on coverage — this measures the condition).

### 2.2 What SURVIVED the meat hunt (certified for the re-seal)

- **Rate leg custody (angle 5)** ✓ [re-executed]: central 60,000 − 3,108 + 30,892 + 4,500 + 800 +
  8 = 93,092 exact; win9 = central − (30,892 − 18,832) = 81,032 exact; win9 18,832 chain =
  MEASURED roundtrip-exact, byte-close-selectable, GATED P8/F8, win5 QUARANTINED with reason;
  LBND2 41,526 confined to the worst-tail row it belongs in; store-nothing = no stored keyframe
  by construction (frame0 = warp of the witness's own render), so no missing keyframe-bytes row;
  rates 0.0619861/0.0539559 reproduce at full precision.
- **Fire-epoch invariance logic** ✓ — a zero-crossing trigger fires at ep675 for ANY positive s*;
  the printed band [6.9e-6, 1.42e-5] is illustrative, s* = 6.8971e-6 sitting 0.04% below its lower
  edge changes nothing (the honest reading is "any s* in the dispute range AND below it").
- **B19's rule-118 + injection-point honesty** ✓ — strengthened since round 1: the sibling's
  probe memo pins the exact `_R` uint8 site in the packet's own inflate.py, and the instrument
  exists; v6's "no dither lever exists in the trainer" claim is correctly TRAINER-scoped.
- **The launcher gate stack as v6's friend** ✓ — flag validation, duplicate-refusal (C13), memory
  preflight (67.61 GiB PASS at v6 scale), throughput gate, governed admission: every v6 safety
  assumption about "riding the launch preflight" has a real, live surface; BLOCKER-1 is about the
  CONFIG artifact, not missing apparatus.
- **Assumption-challenge axis (protocol item 8), stated:** the shared assumption this draft (and
  both seal rounds) operated within is *"the mod32cap 1000-ep trace's absolute-epoch anchors
  (600/650/675/726) transfer unchanged to the 3000-ep run-1 config"* — BLOCKER-1 is that
  assumption surfacing as a measured violation (the config family SCALES epochs; the draft assumes
  ABSOLUTES). Classification: CARGO-CULTED at the config-derivation surface; the fix (absolute
  pins in the new variant) makes it HARD-EARNED.

## §3 VERDICT + COUNTER

**NOT-CLEAN.** 1 BLOCKER (launch-route materialization: measured dry-run REFUSAL on the extras
route + measured wrong-schedule emission on the named-config route; missing build item, F-DET
"0 (config)" under-scoped) + 2 MAJOR (pose block unpinned at the draft surface with #314 unnamed;
locked-mass per-lever coverage unquantified with a 25.3%/22.5×-margin uncovered quarter) + 3 MINOR
(k_max FIN-dwell accounting 6→≤5 · SC-3-ext generation route unnamed · C4 injected knob missing
from the knob tables). All fixes are cheap and none moves the crossing arithmetic, the probe
verdicts, or any gate disposition — but each changes a build item or a consumer-read claim, which
is the bar. Honesty both directions: round 1's arithmetic/provenance certifications all held under
re-attack from different angles; the failures found here live exactly where round 1 did not look
(the draft→launcher materialization seam and the $0-computable coverage table).

**Counter: 0/3.** A v6.2/v7 folding {the crucible-v6 named config build row + pose pin + #314
guard + the §0.3 coverage table + the three minors} restarts the count; rounds after that should
re-verify by RE-RUNNING the launcher dry-run against the NEW config (the measured check is 2
minutes and closes the seam this round opened). NO-OPEN-GATES rule unchanged: the final certifying
round still also waits on P-TAU2 + P-DITHER (sibling in-flight).

Most load-bearing single finding: **BLOCKER-1** — as specifiable today, the launch either gets
REFUSED by the launcher's own confound gate or silently runs τ(675) ≈ 0.886 / Muon@2178, a run the
draft's schedule laws do not describe.

Pointer contest-CPU 0.19110 UNMOVED — this verdict is MEANS.
