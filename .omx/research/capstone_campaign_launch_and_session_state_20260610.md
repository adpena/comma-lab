# Capstone campaign launch + session state (2026-06-10, resume-from-disk)

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

## Harvest contract (next session / wakeup)
Read `experiments/results/capstone_daemon_b16_n100/capstone_result.json` (final) OR
`tail` the daemon log for the live eval_every=5 RD trajectory. Judge: d_seg descent slope + d_pose
hold at the 64 KB budget. Then route per the reactivation ladder above.
