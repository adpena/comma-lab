# Capstone campaign launch + session state (2026-06-10, resume-from-disk)

> ⚠️ **CORRECTION BANNER (2026-06-11, adversarial review):** several POSITIVE claims below are OVERSTATED.
> Per the adversarial-review synthesis (`capstone_adversarial_synthesis_and_honest_corrections_20260611T015018Z.md`):
> (1) "pose crux fix EMPIRICALLY CONFIRMED" → RETRACTED to "plausible, NOT confirmed" (the 0.437 was one bounce
> of an oscillating series; the "shared-FiLM J≈0" mechanism is wrong — separate rgb_0/rgb_1 heads make the
> differential learnable; needs a controlled A/B). (2) "d_seg could keep grinding down" → "CE plateaus ~0.008"
> (geometric decay, ratios 0.49/0.33). (3) Best capstone state recomputes to **S ≈ 1.75 — 9× WORSE than the
> 0.191 frontier** (d_pose ~0.1 alone = pose term 1.0 = 5× the frontier). (4) The "smaller-than-frontier basis"
> thesis FIGHTS the lab's own param↔d_seg curve (need MORE params for lower d_seg); the realistic path is
> frontier-CLASS params (base_ch≈22–24, still in budget) + the factorized `[seg-blob]⊕[pose-store+FiLM]`
> carrier. Read the synthesis memo for the refined plan; the verdict sections below predate the correction.

**Lead:** The exact frontier pointer is **0.19109982 [contest-CPU]** (`b46897267…`, 177,169 B,
recoded-R3 defensive bank) — **UNMOVED**. This session did NOT move it. What it DID: (1) fixed an
apparatus bug (the canonical pointer JSON was null → repopulated from real eval anchors via the
canonical path); (2) closed #79 (packaging lever — a rigorous NEGATIVE); (3) built + validated the
**capstone campaign runner** (the missing #65/#78 actuator) and **launched the first ORIGINAL-basis
viability run** as a local detached daemon.

## The mission framing (unchanged)
- Pointer-moves are the ONLY progress (sub-0.15 non-negotiable; any sub-0.19 = approved progress;
  defensive banks approved). Tools/memos/negatives are NOT pointer moves.
- The mission = our OWN small learned basis (the capstone), NOT post-hoc compression of the
  borrowed/frozen frontier (those levers are now closed — see below).

## What closed this session (all $0, honest negatives + apparatus)
- **#79 packaging lever — CLOSED.** `evaluate.py:63` counts ONLY `archive.zip`'s `st_size`. The
  frontier zip is at the container floor (1-char member `"x"`, STORED, 100 B overhead) AND the payload
  is at the entropy floor (7.9990 bits/byte; every general coder GROWS it). Zero recoverable bytes.
  Rate only falls via a SMALLER PAYLOAD (the capstone). Audit:
  `archive_packaging_byte_audit_20260610T224611Z.md`. Ledger move-row 8.
- **Apparatus:** `canonical_frontier_pointer.json` was null (never written this $0 session) →
  repopulated via `refresh_canonical_frontier_from_local_state` (cpu 0.19109982 `b46897267`,
  cuda 0.20533003 `9cb989cef519`). NOT a hardcoded literal — the canonical scan path.
- The 8 prior no-moves (#64/#69/#71/#72/#73/#54 + packaging #79) jointly prove: **no $0 post-hoc op
  on the frozen 177 KB borrowed frontier lowers rate without an equal/larger distortion penalty.**
  The frozen basis is at its post-hoc floor. The ONLY rate path is a retrained smaller basis.

## The capstone — REAL state (corrects the "blocked on GPU" framing)
The capstone (`src/tac/capstone_vq_nerv/`) is MORE built than "pending" implied: trainer threads the
6 stored GT pose scalars through `_PoseFiLM`; `capstone_byte_budget.py` sweeps base_channels×dtype;
`export.py` builds the int8 archive. The 12-pair smoke (`.omx/tmp/capstone_real_recipe.log`) proved
**d_seg descends 0.5073→0.0117 (43×) on the EXACT scorer** — the inert-loop bug is genuinely fixed.

BUT the 12-pair smoke ran a **big fp32 decoder (407 KB, rate 0.28)** and **pose bounced (d_pose 0.437,
NOT held)**. The summary's "71,968 B / S 0.1355" was a *projection* from the sizing tool, not the
trainer's real output. So the true gap was never "just GPU" — it was: run at the SMALL int8 budget,
with the FiLM actually holding pose, across MANY pairs.

## What was built this session: `experiments/run_capstone_campaign.py` (the #65/#78 actuator)
Thin CLI → `score_aware_loop.targets` (frozen DistortionNet + GT-target cache) → `mlx_pr95_port`
score-bridge → capstone VQ-NeRV+FiLM train → int8 byte-close (100 B-floor ZIP) → exact advisory S.
**MVP smoke validated end-to-end** at base_ch=16 int8: **archive.zip = 64,369 B → rate term 0.0429**
(the sub-0.15-capable budget, now MEASURED on a real export, not projected). Timing: ~2.4 s/pair·epoch
local (torch-CPU scorer is the bottleneck; renderer is MLX-fast).

## DECISIVE viability run — LAUNCHED (local detached daemon, $0, TRUSTED)
- **pid 7692**, log `.omx/tmp/capstone_daemon/capstone_b16_n100_20260610T230515Z.log` (latest path in
  `.omx/tmp/capstone_daemon/LATEST_LOG.txt`), out `experiments/results/capstone_daemon_b16_n100/`.
- Config: **100 pairs, base_ch=16, int8, 100 epochs, descent recipe (muon_lr=3e-2, grad_clip=50),
  eval_every=5.** ~6–7 hr wall. Renderer=MLX, scorer=torch-CPU (TRUSTED per "local CPU + MLX GPU good";
  MPS is NEVER authority — the runner hard-refuses `--device mps`).
- **The decisive question it answers:** at the 64 KB int8 budget, across 100 distinct pairs (NOT
  trivially overfittable like 12), does d_seg descend toward the frontier's 5.6e-4 **AND** does the
  FiLM hold d_pose toward the tube? Watch the eval_every=5 RD trajectory in the log.

## Gating (the firewall — no fake pointer moves)
- This daemon's score is `[macOS-CPU advisory]` — `score_claim=false`, `promotion_eligible=false`. It
  RANKS + GATES; it does NOT move the pointer (CLAUDE.md "Frontier scores are pointer-only").
- **Reactivation ladder:**
  1. Daemon RD curve descends d_seg + holds d_pose at 64 KB on 100 pairs → **viability CONFIRMED**.
  2. Then the FULL 600-pair run is justified — but locally that's ~24 min/epoch = multi-day, so the
     600-pair sub-0.15 candidate is the **paid GPU step** (Modal/Vast CUDA, lane-claimed, <$ budget,
     fail-closed). This is the ONE paid action and it needs operator GPU authorization.
  3. Byte-close the 600-pair archive → **paired contest-CPU + contest-CUDA exact eval** → if recomputed
     S < frontier → `scorer_quotient_candidate_row` (contest_cuda, exact_evaluate) → **pointer move**.
- If the daemon STALLS (d_seg won't descend at 64 KB, or FiLM won't hold pose at scale) → that's the
  real finding at $0: redirect (bigger budget? stronger pose-FiLM? higher pose_weight? per-the
  score-domain Lagrangian) BEFORE any paid run.

## Open in-flight (non-capstone)
- #63 (decisive root-cause test) still flagged in_progress — re-interpret through the #76 fixed loop
  next harvest; likely subsumed by the working loop + this runner.
- #27 (HiNeRV B-lane) flagged in_progress — separate lane, not on the capstone critical path.

## CRUX FIX (2026-06-10, commit `11f15a56d`) — pose-FiLM was SHARED, now PER-FRAME
The capstone's `_decode_with_film` applied ONE shared FiLM to the common feature, then decoded both
frames from it — but PoseNet scores the frame0↔frame1 DIFFERENTIAL (ego-motion), so the shared FiLM had
~0 Jacobian in the rewarded pose direction (`J_scorer·J_renderer ≈ 0`) → d_pose bounced 0.437. Fixed:
per-frame `film0`/`film1` modulate the feature DIFFERENTLY before each rgb head (matching #84 that held
d_pose 2.7e-4). Identity-init preserved (smoke: max|Δ|=0.0), per-frame control verified (perturb film0 →
frame0 Δ=5.58, frame1 Δ=0.000), FiLM rides in `decoder_weights` (+0.2 KB), 22 tests pass incl. the
real-scorer pose-hold + a new shared-FiLM-regression guard. Buggy daemon killed; corrected daemon
relaunched at `experiments/results/capstone_daemon_b16_n100_perframe/`.

## v2 levers PRE-REGISTERED (queued for the harvest decision — keeps the next unit crisp)
1. **Score-domain pose loss (sqrt, not MSE).** The score's pose term is `√(10·d_pose)`, whose gradient
   carries a factor `5/√(10·d_pose)` that EXPLODES as d_pose→tube. The capstone minimizes MSE pose-loss
   at fixed `pose_weight=1.0` → it OVER-weights pose early (where the score barely cares) and UNDER-weights
   near the tube (where the score cares enormously). v2: pose loss = `√(10·d_pose)` directly (the goal's
   `α·B+β·d_seg+γ·√d_pose`). NOTE: for a FiLM HANDED the answer, pose is a conditioning not a capacity
   problem, so the crux fix is primary; sqrt-loss is the refinement if d_pose holds-but-not-tightly.
2. **Factorized two-blob carrier (the redirect IF the single decoder can't do both at 64 KB).** Lever B
   proved a 64 KB blob hits the SegNet 600-argmax partition (d_seg 0.008) but collapses pose; the crux fix
   makes pose a controllable 6-scalar FiLM. So the sub-0.15 carrier may be the explicit factorization
   `[seg-argmax blob] ⊕ [6-scalar pose-FiLM]` — each scored quantity in its OWN minimal representation —
   rather than one shared RGB renderer. Build this if the 100-pair run holds pose but stalls d_seg at 64 KB.

## CONFIG: the sub-0.15 capacity ladder (MEASURED byte budget, `capstone_byte_budget.py`)
int8 600-pair archive rate by base_channels (sub-0.15 rate budget if d_seg→5.6e-4 + d_pose→tube is
**< 0.077 = 115,640 B**):
| base_ch | dec_params | total_B | rate | sub-0.15? |
|---|---:|---:|---:|---|
| 16 | 59,608 | 72,014 | 0.0479 | YES (conservative floor) |
| **20** | **85,125** | **97,025** | **0.0646** | **YES — the sub-0.15 CAPACITY CEILING** |
| 24 | 114,934 | 126,690 | 0.0844 | no (overshoots budget) |
| 36 | 231,783 | 242,448 | 0.1614 | no |
**Decision (evidence-driven):** the decisive viability test is the LARGEST sub-0.15-capable config —
**base_ch=20** (+43% decoder capacity vs 16) — because the hard target d_seg→5.6e-4 came from a 162K-param
decoder; if even base_ch=20 (85K) can't reach it, the architecture is the ceiling; if it does, sub-0.15 is
PROVEN and we optimize *down* to 16 for lower rate. Switched the running daemon 16→20 (only ~8 min in;
GT cache persists). base_ch≥24 overshoots the byte budget. This is the capacity refinement of v2 lever 1.

## INFLATE RUNTIME BUILT — the contest exact-eval path is UNBLOCKED (commits `60b1d6635` + `dfb1cb4fe`)
The MLX→numpy portability contract + contest inflate now exist and are score-parity-verified:
- `src/tac/capstone_vq_nerv/numpy_reference.py` — pure-numpy port of `_decode_with_film` (incl per-frame
  FiLM), parameterized over base_channels. Parity vs the MLX render: **d_seg |Δ| = 0.0 (EXACT)**, d_pose Δ
  = fp16 pose-store roundtrip (rel 5e-5, argmax-invariant). No MLX/torch (runs on Linux/CUDA).
- `src/tac/capstone_vq_nerv/inflate.py` (≤70 LOC) + `runtime/inflate.sh` — parse archive → numpy decode →
  write `(N,874,1164,3)` uint8 frames `TensorVideoDataset` reads. No scorer at inflate.
- **NO-FAKE find + fix:** the prior export archived only `decoder.parameters()`, DROPPING the per-frame
  FiLM weights + pose-norm stats the render depends on (a decoder-only archive is un-inflatable for a FiLM
  bundle). The runner (`run_capstone_campaign.py::_export_int8_archive`) now uses
  `full_render_weights_from_bundle` (decoder+FiLM, contest naming) + a `capstone_config_v1` sidecar
  (pose_mean/std/base_channels). **Verified roundtrip: runner export → inflate.py → (8,874,1164,3) frames.**
- ⟹ the chain **trained bundle → byte-closed contest-inflatable archive → inflate → evaluate.py** is
  CLOSED. The ONLY remaining blockers to a pointer move are (1) viability confirmation (the running daemon)
  and (2) the paired contest CPU+CUDA exact eval (Modal, paid). The inflate is no longer a blocker.
- NOTE: the running daemon (pid 26696) loaded the OLD runner export (uninflatable) — fine, its value is the
  d_seg/d_pose trajectory + byte count, not inflation; the 600-pair CONTEST candidate run will use the
  fixed runner and produce an inflatable archive.

## RUNNER FIX: streaming telemetry (commit `39ad7752d`) — prior daemons ran BLIND
`CapstoneTrainer.train()` returns the trajectory only at the END (no per-epoch print), so the 100-pair
daemons logged nothing for ~10 hrs — useless for mid-run harvest (a "Max observability" violation). Fixed:
the runner passes a `_StreamingTelemetry` that writes each eval row to `<out>/trajectory.jsonl` + stdout.
ALSO right-sized the viability run: **48 pairs** (≈2× faster epochs than 100, still non-trivially-
overfittable) + eval_every=2 → first RD point ~6-8 min, clear curve ~30-40 min. Current daemon:
**pid 46817, base_ch=20, 48 pairs, eval_every=2, crux-fixed per-frame FiLM**, out
`experiments/results/capstone_daemon_b20_n48_stream/`.

## ★ VIABILITY SMOKE RESULT (2026-06-11T00:38Z, base_ch=20, 48 pairs, crux-fixed per-frame FiLM)
Full RD curve (eval_every=2, streamed): `experiments/results/capstone_daemon_b20_n48_stream/trajectory.jsonl`
| epoch | 2 | 4 | 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|---|
| d_seg | 0.2754 | 0.0248 | 0.0151 | 0.0129 | 0.0123 | 0.0113 | **0.0106** |
| d_pose | 17.81 | 0.243 | 0.096 | 0.335 | 0.062 | 0.207 | **0.140** |

**VERDICT: crux fix CONFIRMED · d_seg INCONCLUSIVE (NOT a kill — needs a long train).**
- **Pose crux fix EMPIRICALLY CONFIRMED:** d_pose 135→~0.1 (the old shared-FiLM stalled at 0.437). The
  per-frame FiLM controls the frame0↔frame1 motion. It oscillates 0.06–0.34 (needs the v2 √-loss/EMA/LR
  stabilization to lock to the tube, but it's structurally solved). **This validates the whole turn's
  central fix.**
- **d_seg: 0.505 → 0.0106 in 14 epochs, still descending (~0.0007/2ep) but slowing (~14× slower than the
  initial drop).** At 14 epochs the seg term is ~1.06 — far above sub-0.15's 0.056. BUT: the frontier's
  d_seg 5.6e-4 came from PR95's **29,650-epoch** 8-stage curriculum; 14 epochs is nothing. Declaring a
  "seg-capacity wall" here would be a Catalog #307 error (paradigm KILL from an under-trained
  implementation). The honest state: **the d_seg asymptote is unresolved** — it could keep grinding down
  over thousands of epochs (under-training) OR plateau ~0.008–0.010 (capacity). The slowing rate leans
  toward a capacity concern, but 14 epochs cannot decide.
- **Pipeline fully validated:** GT-targets → MLX train → per-frame-FiLM render → int8 byte-close →
  contest-inflatable archive → (numpy inflate, d_seg-exact) is end-to-end correct + observable.

**ROUTING (the decisive next test):** a LONG train (≥1000s of epochs, PR95-class curriculum) to resolve
the d_seg asymptote. This is the campaign (#65). Local MLX is ~2–3 min/epoch (48 pairs) → 1000 epochs ≈
40 hrs (a multi-day detached daemon, $0) — OR the numpy reference now enables a **torch/CUDA port** for
fast Modal training (the MLX→numpy→torch portability path). **This is the real fork: invest a multi-day
local run or a funded CUDA port to find whether the smaller basis can reach frontier-class d_seg.** If it
plateaus high → the smaller-basis-for-rate path is genuinely walled (frontier is near-minimal, confirming
#71) and sub-0.15 needs lever A (class-shift). If it reaches ~5.6e-4 at the 97 KB budget → sub-0.15
candidate → contest eval → pointer move.

## Harvest contract (next session / wakeup)
**`tail experiments/results/capstone_daemon_b20_n48_stream/trajectory.jsonl`** (live per-epoch RD curve)
OR read `…/capstone_result.json` (final). Latest log: `.omx/tmp/capstone_daemon/LATEST_LOG.txt`.
**Decisive read: does d_pose now HOLD/descend toward the tube (crux fixed) vs bounce ~0.4 (old shared
FiLM)?** Route: (a) d_pose holds + d_seg descends → fund the 600-pair candidate (needs the MLX→torch port
for CUDA, OR a multi-day local MLX run) → byte-close → paired CPU+CUDA exact eval → pointer move; (b)
d_pose holds but d_seg stalls at 64 KB → v2 lever 1 (sqrt-loss) then lever 2 (factorized carrier); (c)
d_pose still bounces → the crux fix is insufficient, escalate the pose-control mechanism.
