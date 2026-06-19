---
title: "DEEP ADVERSARIAL APPARATUS AUDIT — the TERMINAL FINDING is RE-OPENED; the 0.191 frontier is BORROWED, our own full-stack PR95 was never trained at scale, and the 'walls' are measured on a throughput-starved / stage-2-stalled / borrowed stack — APPARATUS ARTIFACT, not physics"
authority: "[contest-CPU advisory / measured-synthesis] — pointer UNMOVED 0.19110; $0; NO paid dispatch; NO score claim"
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
pointer_moved: false
date: 2026-06-19
verdict: RE_OPENED_APPARATUS_IS_THE_BREAKTHROUGH_BLOCKER_THE_CORRECTED_FULL_PR95_CURRICULUM_AT_SCALE_WAS_NEVER_RUN
operator_directive: "I am surprised we haven't broken through optimizing full-stack PR95 — there is no way it is optimal; I think our APPARATUS may be preventing a breakthrough."
re_opens: .omx/research/TERMINAL_FINDING_representation_axis_sub015_exhausted_20260619.md
cross_refs:
  - .omx/research/SESSION_SYNTHESIS_SoT_20260617_20260618.md          # already names this #1 suspect (line 11)
  - .omx/research/pr95_seg_convergence_mechanism_and_recipe_gap_audit_20260611.md  # BUG-A + "prior capacity verdicts IMPLEMENTATION-LEVEL FALSIFIED"
  - .omx/research/generative_axis_dseg_core_design_20260619T004600Z.md  # where 29.3*params^-0.71 was fit (factored-LF/NCA, NOT PR95)
  - .omx/research/generative_axis_nca_amortized_capacity_break_RED_20260619.md
  - .omx/research/dseg_side_feasibility_corners_verdict_20260619.md
  - .omx/research/why_have_we_not_produced_original_frontier_score_meta_diagnostic_synthesis_20260529.md  # "EVERY anchor is a DERIVATIVE of PR101/106"
  - .omx/research/leapfrog_pr112_absorb_recode_verdict_20260610.md     # the frontier = lossless recode of borrowed substrate
---

# Deep adversarial apparatus audit — VERDICT: **RE-OPEN.** The terminal finding's walls are an APPARATUS ARTIFACT.

**Operator hypothesis CONFIRMED by measurement.** The "representation-axis sub-0.15 exhausted / frontier
~0.191 is the real floor" terminal finding rests on three walls (capacity power law, survival, rate/d_seg
tension) that were **all measured on a training stack that is either BORROWED, throughput-starved, stalled
in stage 2 of 8, or fit on the WRONG (tiny) architecture.** The decisive fact: **the corrected full-stack
PR95 8-stage curriculum at n600 — the exact thing the operator named — was NEVER run to completion. NO paid
GPU retrain was EVER dispatched, despite the bug being fixed 8 days ago and the cost being scoped at
$12–49.** Every "wall" verdict measured something other than an optimally-trained PR95 HNeRV decoder. All
`[contest-CPU advisory]`; pointer UNMOVED 0.19110; no score claim, no paid spend this unit.

---

## 0. The headline (read this first)

| Terminal-finding claim | What it was ACTUALLY measured on | Verdict |
|---|---|---|
| **The 0.191 frontier is "near the real floor"** | a **BORROWED** substrate: PR101 (@SajayR public) decoder weights + PR95 arch + PR112 (@mattneel public) entropy coder, lossless-recoded. **0% of its d_seg/distortion is ours-trained.** | the floor is a *competitor's* result, not a measured limit of OUR training |
| **capacity wall `d_seg ∼ 29.3·params^−0.71`** | `factored RANK-1 LF` (a deliberately-narrow d_seg-only decoder, bc12≈few-K params) + NCA gates (10K params) — **NOT the PR95 229K HNeRV decoder** | extrapolation from the WRONG architecture class; the real PR95 decoder beats the power-law prediction at 120 trivial epochs |
| **survival wall (realized d_seg ~0.0067)** | **flat-painted** partition stores / curve-cores (no interior texture) | a wall on FLAT painting; the frontier's own move (learned continuous texture) is exactly what was never trained |
| **our own full-stack PR95 "stuck at d_seg≈0.50"** | a curriculum throttled by **BUG-A** (muon_lr 0.03→2e-4 = 150× too small + 100% grad-clip) — froze d_seg at the init value | a RECIPE BUG, self-identified 2026-06-11 as "IMPLEMENTATION-LEVEL FALSIFIED" |
| **the corrected curriculum reaches the basin?** | **NEVER MEASURED at scale.** Best n600 run: 22.7 hrs CPU → epoch 5,793/50,000, **STILL IN STAGE 2 of 8** (never reached the Muon-finetune d_seg-finishing stages) | the decisive measurement does not exist |

**VERDICT: RE-OPEN. The apparatus IS the breakthrough blocker.** The single likeliest frontier shift: **run
the BUG-A-corrected full 8-stage PR95 curriculum at n600 on a paid GPU (the de-risked, scoped, never-fired
dispatch) and measure whether d_seg crosses below the borrowed frontier's 0.00056.** Every prior wall was
measured on a stack that could not have reached it.

---

## 1. SUSPECT #1 (CONFIRMED) — the frontier is BORROWED; our own training never reached it

**The 0.19110 frontier (`lane_pr110_payload_entropy_recode`) is a lossless entropy RE-CODE of a borrowed
substrate, not an OUR-trained vehicle.** Measured provenance chain (file-cited):

- Lane name (`.omx/state/lane_registry.json`): "PR110 payload entropy recode **(absorb PR112)**".
- `leapfrog_pr112_absorb_recode_verdict_20260610.md`: the win is **rate-only, 0% distortion** — recoded raw
  decode is **BYTE-IDENTICAL** to R3 (`dacf6b33… == dacf6b33…`); `d_seg 0.00056` and `d_pose 2.94e-05`
  **unchanged by construction**. The entropy coder is "vendored verbatim from PR #112's `codec_ctx.py`".
- `public_pr112_frontier_beat_intake_20260610.md`: lineage = "PR #101 (**@SajayR**) content; PR #110 (us)
  selector + inflate; **PR #95 arch**; PR #98 channel bias". PR112 author = **mattneel** (external).
- `why_have_we_not_produced_original_frontier_score_meta_diagnostic_synthesis_20260529.md` (line 82):
  *"EVERY single one is a derivative bolt-on of PR101 (codec.py grammar) or PR106 (latent sidecar)."* Best
  ORIGINAL class-shift = CPU **0.198696 (3.5% ABOVE the frontier)**.

**Why this is the load-bearing crack:** the terminal finding says "the frontier is near the real floor." But
the frontier's *distortion* (the d_seg/d_pose that the walls are about) is a **competitor's trained result we
absorbed** — we hold the bytes, not the training. So "we can't beat 0.191" was never tested by *training* —
it was tested by *shrinking/re-coding a borrowed decoder* (re-pack DEAD, bit-shrink CAPS, deletion ~0). Those
3 REDs prove the **borrowed decoder is task-dense** — they say NOTHING about whether an OPTIMALLY-trained
PR95 decoder (ours, from scratch, full recipe) lands above or below 0.191. The terminal finding conflated
"the borrowed frontier is saturated" with "training cannot beat it." Those are different claims; only the
first is measured.

---

## 2. SUSPECT #2 (CONFIRMED) — full-stack PR95 was NEVER trained optimally; the corrected run-at-scale does not exist

### 2a. BUG-A: the curriculum silently throttled the optimizer 150× (self-identified, then NOT re-run at scale)
`pr95_seg_convergence_mechanism_and_recipe_gap_audit_20260611.md` is the repo's OWN diagnosis: the
`configure_stage` path **overwrote the working `muon_lr=0.03` with PR95's torch stage-8 value `2e-4`** (150×
smaller) **and forced `grad_clip_muon=1.0` (100% clip every epoch)**. The decisive A/B (same arch, same loss,
same 15 epochs, same init 0.50727):

| arm | muon_lr | d_seg after stage 1 (CE, 15ep) | |
|---|---:|---|---|
| BUGGY | 2e-4 | **0.50727 — FROZEN at init (0% movement)** | the "d_seg≈0.50 wall" |
| FIXED | 0.03 | **0.06647 (7.6× descent)** → 0.0165 (stage 2) → **0.0120 (stage 3)** | the wall dissolves |

That memo states verbatim: *"The prior 'capacity-limited' verdicts that rested on the buggy curriculum
(c1prime 0.0097) are **IMPLEMENTATION-LEVEL FALSIFIED per Catalog #307** — they measured the throttled
recipe, not the architecture's floor."* And it names **"THE NEXT STEP … the CORRECTED full 8-stage
curriculum at n600 … on a PAID GPU … the exact-row campaign the operator's sub-0.15 goal points at."**

### 2b. That paid run was NEVER dispatched (measured from git + dispatch claims)
- Bug fixed **2026-06-11** (commit `f6a913ccc`). Readiness scoped: `15dcc1739` priced it (T4 $0.59/A10G
  $1.10/A100 $2.10/hr; **compressed 3–10k epochs ≈ $12–49**; packet "armed-not-fired" per `ebad06476`).
- **Every dispatch claim 2026-06-09→onward is `local_mlx` / `local_macos_mlx` / `local_pipeline_build`** —
  research-signal only. **ZERO paid Modal/Vast/Lightning PR95 retrain ever fired.** (grep of
  `active_lane_dispatch_claims.md` + `git log --since=2026-06-11 | grep -iE 'modal|vast|lightning|paid'`.)
- Instead, the campaign (2026-06-12→19) pivoted entirely to **tiny $0 CPU feasibility gates** — factored-LF,
  curve-core, flat-NCA, amortized-NCA, sub-pixel/warp, int5-PTQ, task-space — none of which is the PR95
  decoder, and concluded "terminal."

### 2c. The ONE n600 PR95 run that exists is throughput-starved + stage-2-stalled (measured)
`experiments/results/bindall_arm_b_canonical50k_mh_n600/torch_vehicle_summary.json`:
- `device: cpu` (MPS has a "prohibitive ~3min warmup" and is NEVER authority → the d_seg-critical training
  runs on the **SLOWEST** substrate, ~14.1 s/epoch).
- **22.7 HOURS wall-clock → global_epoch 5,793 / 50,000 budget = 11.6%**, and `stage_name: stage2_v331_softplus`
  — **STILL IN STAGE 2 of 8.** The full 29,650-ep curriculum on CPU = **~4.9 days**.
- It **never reached stages 5–8** (C1a, sigma, **Muon-finetune**) — which the 06-11 mechanism memo identifies
  as where the d_seg heavy-lifting *finishes* below CE's floor. Best d_seg 0.00353 is a **mid-stage-2,
  early** number, not a converged one. The other n600 runs (`corrected_n600`, `fullstack_n600`,
  `canonical_n600`) all DIED at epoch 20–90 (stage 1 only).

**This is the apparatus bottleneck in one sentence:** the d_seg-critical training is pinned to a $0 local CPU
loop that takes ~5 days to run the curriculum once, so it is never run to completion — and the "walls" are
read off its first 11%.

---

## 3. SUSPECT #3 (CONFIRMED) — the "capacity wall" is fit on the WRONG architecture; the real decoder beats it

The power law `d_seg ∼ 29.3·params^−0.71` was fit (`generative_axis_dseg_core_design_…md`,
`factored_lf_core_capacity_gate_…`) on **`factored RANK-1 LF`** (a deliberately-narrow, d_seg-only learned
decoder; bc12 ≈ few-K params) and the **NCA gates** (10K–17K params). It was then extrapolated ("frontier-grade
needs ~10.7M params / ~628K params → rate forfeits"). The recursive review itself flagged this:
*"factored-LF '10.7M params' is a 2-point extrapolation ±a decade."*

**The real PR95-class decoder refutes the power law directly (measured, capacity ablation 2×2,
`capstone_capacity_ablation_2x2_20260611`, 120ep CE-ONLY — just stage 1):**

| config | params proxy | d_seg @ 120ep CE-only | |
|---|---|---:|---|
| bc20, 48 pairs | base_ch=20 | **0.00376** | |
| **bc24, 48 pairs** | base_ch=24 | **0.00285** | **MORE capacity → 24% LOWER d_seg** (wall says it should barely move) |
| bc20, 192 pairs | base_ch=20 | 0.01920 | (more data ≠ better at fixed tiny epochs — data/epoch trade, not capacity) |

At **120 trivial CE-only epochs** (no tau_softplus refine, no l7, no Muon-finetune — i.e. NONE of the
d_seg-finishing curriculum), bc24 already reaches **0.00285 — only ~5× the frontier's 0.00056.** The power
law predicts the wall is at the params axis; the measurement shows (a) adding capacity lowers d_seg
(contradicting a wall) and (b) the run is nowhere near converged (stage 1 of 8, 120 epochs vs 29,650).
**The capacity wall is an artifact of fitting a curve to under-trained, under-capacity, wrong-architecture
points.** Per the memory anchor: bc24's RATE floor S=0.1353 IS sub-0.15-marginal **if d_seg < 0.000147** —
which a fully-trained curriculum (not a 120-ep CE smoke) is the test of, and which was never run.

---

## 4. SUSPECT #4 (PARTIAL) — meta-layer/apparatus suppression channeled effort away from the one real test

Not a code gate that hard-blocked the dispatch — a **process** suppression, which is exactly the
"UNIQUE-AND-COMPLETE-PER-METHOD / apparatus-blind-to-shared-assumption" failure class CLAUDE.md warns about:

- The 270+-gate / $0-feasibility-gate culture made it *cheap and rewarded* to run dozens of tiny CPU probes
  (each producing a clean "RED" memo) and *expensive/uncomfortable* to fire the one $12–49 paid run that
  would actually answer the question. The campaign optimized for **measured-RED memos** (means) over the
  **one measured exact row** (end) — the precise means/ends inversion the GOAL firewall names.
- The "MPS is never authority" + "MPS ~3min warmup" discipline (correct for *scoring*) was over-applied to
  *training throughput*, pinning the gradient loop to CPU and making the curriculum effectively un-runnable
  locally — so it was never run, rather than dispatched to a GPU.
- The terminal finding then **canonicalized the under-measurement as physics** ("S_floor 0.11797 REFUTED";
  "near the real floor"), which would have permanently closed the highest-EV path on artifact evidence. The
  SoT's own re-open header (line 11) already names this exact suspect — the apparatus produced a "terminal"
  conclusion the apparatus's own earlier memo (06-11) had pre-falsified.

---

## 5. The ranked apparatus blockers (by frontier-shift EV)

1. **[HIGHEST EV] The corrected full PR95 8-stage curriculum at n600 was never run to convergence on a GPU.**
   Fix: fire the scoped, de-risked, armed-but-never-fired paid dispatch (BUG-A-corrected `muon_lr`/clip,
   8 stages, n600, EMA-warmup, eval_roundtrip, the live-SegNet #76 loop). $0-test surrogate first (below).
2. **The "capacity wall" power law is fit on factored-LF/NCA, not the PR95 decoder.** Fix: re-fit (or
   discard) the power law using the ACTUAL HNeRV decoder at {bc20, bc24, bc28} *fully trained*, not tiny
   d_seg-only cores at 120 CE epochs. bc24 already shows capacity LOWERS d_seg.
3. **The frontier is borrowed; "can't beat 0.191" was tested by re-coding it, not by training.** Fix: treat
   the borrowed 0.191 as the BAR to beat with an OWN-trained decoder, not as a measured floor.
4. **Training throughput is pinned to local CPU (~5 days/curriculum) → the decisive run never completes.**
   Fix: GPU dispatch (the throughput, not the method, is the gate — `22c483f74` even says "accelerator not
   gate" but then the accelerator was never used).
5. **[PROCESS] The $0-gate culture rewards RED-memo volume over the one exact row.** Fix: per GOAL firewall,
   the next unit must be the exact-eval-feeding dispatch, not another feasibility gate.

---

## 6. The single likeliest frontier shift + the $0-or-cheap test that measures it

**The shift:** an OWN-trained PR95/HNeRV decoder (bc20 or bc24, full corrected 8-stage curriculum) whose
converged d_seg lands near/below the borrowed frontier's 0.00056 at well-below-frontier bytes (bc24 rate
floor 0.1353; bc20 0.1178) → projects **sub-0.15** if d_seg reaches ~1.5e-4 (bc24) / ~3.2e-4 (bc20).

**The cheap test ladder (measurement-first, de-risks the paid run):**
- **$0, days, local-CPU-resumable (already partly running):** let `bindall_arm_b_canonical50k_mh_n600`
  actually REACH stages 5–8 (it's only at stage 2 after 22.7h). Per the long-resumable-sweeps directive,
  keep it running with per-stage checkpoints; read d_seg at the Muon-finetune stage — the stage the
  mechanism memo says finishes the descent. The current "0.0035 cap" is a stage-2 number, not a verdict.
- **~$0.30, the dominant-uncertainty collapse (per `15dcc1739`):** a GPU **step-time smoke** — measures
  s/epoch on T4/A10G so the full-curriculum cost + wall-clock is known, not guessed.
- **~$12–49, the decisive exact row (the never-fired dispatch):** corrected full 8-stage curriculum at n600
  on T4/A10G → byte-close (PR101 grammar, already built per G3) → `upstream/evaluate.py` CPU+CUDA. **This is
  the one measurement that converts "terminal" into a real verdict.** It either (a) lands sub-frontier d_seg
  → the breakthrough, or (b) confirms a REAL capacity/convergence wall on the actual decoder (then the
  terminal finding is earned, not assumed).

**CONFIRM-or-REOPEN: RE-OPEN.** The terminal finding is **NOT confirmed** — its walls are measured on a
borrowed substrate (suspect 1), a 150×-throttled then never-rerun-at-scale curriculum (suspect 2), and a
power law fit on the wrong tiny architecture (suspect 3), with a process that channeled effort away from the
one real test (suspect 4). The apparatus IS the breakthrough blocker. The corrected full-stack PR95 retrain
at scale is the highest-EV unfired shot and must be the next unit.

---

## 7. NO-FAKE ledger (MEASURED vs INFERRED)

- **MEASURED (read directly off files this unit):** frontier provenance = borrowed/recoded (lane registry +
  3 memos, byte-identical recode); BUG-A A/B (0.50727 frozen vs 0.0120 fixed); the n600 50k run is at
  epoch 5,793/stage-2 after 81,888 s CPU (summary.json); capacity ablation bc20 0.00376 vs bc24 0.00285 at
  120 CE-ep (result JSONs); NO paid PR95 dispatch in git/claims since 06-11; cost scoping $12–49 (commit).
- **INFERRED (reasoning, not a measured row):** that the corrected full curriculum at n600 WILL reach
  sub-frontier d_seg — this is the HYPOTHESIS the never-fired run tests, NOT a claim. The bc24-sub-0.15
  projection depends on d_seg<1.5e-4, unproven. The capacity power law is *refuted as fit* (wrong arch) but
  the true PR95-decoder d_seg(params) curve is not yet measured to convergence.
- **NOT claimed:** no score moved; pointer UNMOVED 0.19110; no promotion; no exact row produced this unit.

## Observability surface
Every claim above is anchored to an absolute file path + the measured field (summary.json `last_eval`/
`global_epoch`/`stage_name`, result JSON `d_seg_final`, lane_registry name, the three provenance memos, git
commit shas, dispatch-claim rows). Axis `[contest-CPU advisory]`, score_claim=false, pointer_moved=false.

## Canonical-vs-unique decision per layer
This is an audit memo (no new code/substrate). Frontier-provenance read, BUG-A A/B, capacity-ablation, and
throughput arithmetic all REUSE existing measured artifacts (ADOPT_CANONICAL — apples-to-apples with the
campaign's own runs). The verdict logic (CONFIRM/RE-OPEN) FORKS only in re-weighting the same measurements
against the terminal finding's claims.
