# #205 UPDATED launch-config recursive adversarial review (post-OOM-fix + store-nothing carrier)

**Axis:** `[macOS-MLX / macOS-CPU advisory] NON-PROMOTABLE`. No score/frontier/promotion claim. Pointer **0.19110 UNMOVED**.
**UTC:** 2026-07-02T23:10:03Z · **git HEAD:** `ccfeccc1b` (reviewed files unchanged since; `5b4d2c9ed` OOM-fix + `dd3aa824f` store-nothing render + `0357fb306` triality) · **reviewer:** recursive-adversarial (AXIS-9, review-only, CONTAINMENT — no GPU fired).
**Mandate:** the LAST SEAL failed on TWO blind axes (OOM-death + unmeasured naive pose). This review is MEASURED, not reasoned — every claim below was EXECUTED (dry-runs, memory-preflight, powerplay CorrectnessDemonstration, test suites) at the REAL config, or is explicitly labelled a projection.

---

## VERDICT: **PROCEED-WITH-REVISIONS** (a MEASUREMENT run, not a sealed sub-0.19 candidate)

The config is **runnable** (measured-component projection SAFE + self-protected), **byte-close BIT-EXACT** (both carriers), **flag-clean** (0 invented), **coherent + deterministic + resumable**. It is NOT sealable on the scored quantities, and it MUST NOT be — d_seg and d_pose are the run's DELIVERABLES, unmeasured on a trained witness. The pose is **OPEN** on the SDF witness. **Launch it as a MEASUREMENT run only after the R1 revision (a small trained store-nothing pose smoke) confirms the trained d_pose actually closes through the real decode — that is the operator HOLD condition (1), and it is cheap.** Full unconditional PROCEED/SEAL is NOT warranted while the pose close is a projection.

---

## Round 1 — Runnability (AXIS-9a): CONFIRMED SAFE (measured-component projection + self-protected)

MEASURED (executed `launch_witness_run.py --dry-run` for BOTH arms + standalone `witness_memory_preflight.py --strict`):

| arm | flags valid | mem-preflight | projected peak | safe ceiling | tag |
|---|---|---|---|---|---|
| `sealed_205` (table) | **83/83** | rc=0 | **67.61 GiB** | 89.6 (70%×128) | **SAFE** |
| `store_nothing_205` | **84/84** | rc=0 | **67.61 GiB** | 89.6 | **SAFE** |

- Breakdown: fixed 15.0 + **cf_mx_cache 43.2** (the dominant RESIDENT term, self-orient 0.072 GiB/pair @ n600) + gt 3.41 + verdict 6.0.
- **CONFIRMED the OOM fix mechanism.** The launch.sh does NOT emit `--verdict-batch`; the trainer argparse default is **32** (verified `train_levelset…py:3604`), and the preflight `DEFAULT_VERDICT_BATCH=32` matches → both parse verdict_batch=32 → chunked → the +66 GiB verdict spike is bounded to a ~6 GiB floor. `_verdict_dseg_dpose_chunked` runs under `torch.inference_mode()` eval-mode BN (running stats = batch-independent) + per-pair-independent argmax/MSE → **bit-identical d_seg, ~1e-6 BLAS d_pose** (already MEASURED in the OOM ledger: 70.65→10.03 GiB, d_seg identical). The verdict is **PURELY OBSERVATIONAL** (trainer line 2427-2436: "the training loop NEVER reads its result") → chunking is score-neutral **by construction**, not by a GPU re-run diff (which would be confounded by MLX-GPU cross-process nondeterminism).
- **Self-protection VERIFIED:** the launcher step (b1) REFUSES (rc=4) a config whose projected peak > 0.70×RAM; `--verdict-batch 0` at n600 → 127.6 GiB → REFUSE. The gap the failed SEAL left (B=8 throughput bench never projected MEMORY at n600) is closed.

**CONCERN C1 (defense-in-depth, non-blocking):** the safety depends on TWO implicit defaults staying coupled — the trainer argparse default (32) AND the preflight `DEFAULT_VERDICT_BATCH` (32) — because launch.sh is SILENT on `--verdict-batch`. If a future edit changed the trainer default to 0 (unchunked), the preflight would STILL assume 32 → say SAFE while the real run OOMs. This is the "implicit-default / comment-only-contract" antipattern. **REVISION R2:** emit `--verdict-batch 32` EXPLICITLY in the sealed/store-nothing configs so launch.sh is self-documenting and the preflight parses the REAL value. (Accepts a break of §7 byte-identity — see C4.)

**CONCERN C1b:** the 67.6 GiB is a **measured-component projection**, NOT an end-to-end measured peak_rss of the actual n600 training loop (the run OOM-died before completing; containment forbids running it now). The projection is grounded in N=600 micro-probe MEASUREMENTS of the two dominant carriers (verdict 66→6, cf_mx_cache 0.072/pair) + a conservative 15 GiB fixed overhead — materially stronger than the failed SEAL's B=8 surrogate — but the true end-to-end n600 peak remains UNMEASURED. `safe_run --rss-cap-mb 90000` is the runtime backstop; the run MUST be watched at the first stage boundary (RSS confirm vs projection).

## Round 2 — Scored quantities (AXIS-9b): pose is OPEN + PRE-RESIDUAL; no S is claimed (CONFIRMED HONEST)

Brutally honest, measured state:

- **rate — MEASURED bit-exact through the real decode (n6/t1, `frame0_max_abs_uint8_diff=0`):** store_nothing pose-carrier section **1049 B → rate_term 0.0491**; table/real_keyframe ds4 **697941 B → rate_term 0.5133** (native lossless = 6.38). ⇒ the **table arm is rate-DOOMED by itself** (0.51 ≫ 0.19); **store_nothing (0.049) is the rate-viable frontier arm.** Grade = `macos-cpu-advisory-through-decode` (valid for a LOCAL_SEAL). CAVEAT: n6 SMOKE; n600 rate is a projection but store_nothing clearly wins (H+xi = 12 B/pair, H derived FREE).
- **d_pose — OPEN + UNMEASURED on a trained witness.** BOTH carriers on the untrained t1 smoke are POSE-BLIND (store_nothing ≈189.65 ≈ null; table 172.66). The trained residual-closed d_pose is a **PROJECTION** (Track B classmean **pre-residual proxy 4.97**; contribution √(10·4.97)≈7.05, catastrophic). The store-nothing close is contingent on BOTH (a) the trained dxi residual AND (b) the trained witness render being **PoseNet-legible under warp** — the OPEN read-back risk (the table arm sidesteps (b) by warping a REAL keyframe; store_nothing BETS a trained render suffices). **This bet is UNMEASURED.**
- **d_seg — TRAINING gap, unmeasured on this config.** AA real-frame ceiling 0.00086 < need-band; proven arm (mod-dim ≠ 32 witness) reached 0.003698. mod-dim 32 (SEALED Q4) covers composite m~13 with headroom; 19's neutrality is UNMEASURED. d_seg is the BINDING term and the run's deliverable.

**Executed CorrectnessDemonstration (`tac.witness_dsl.powerplay`, fail-closed axis-9):**
- **DEMO 1 (planned launch, best current evidence):** LOCAL_SEAL `accepted=False` — d_seg (PREDICTED) + d_pose (PREDICTED) are SURROGATE violations; pre-residual d_pose 4.97 → S=**7.47** (catastrophic regression). ⇒ the config is **CORRECTLY unsealable on scored quantities** pre-run. This is the honest state, not a defect.
- **DEMO 2 (anti-#205 regression guard):** an ANCESTOR d_pose 3.4e-5 is **REFUSED** (grade `ancestor` surrogate). The exact bug the last SEAL made (quoting the ancestor-RGB number) is now structurally caught.
- **DEMO 3 (illustrative post-run):** even a hypothetical residual-closed store_nothing (d_pose 0.0018, d_seg 0.0012) → S=**0.303**, still > 0.191. Sub-0.19 with store_nothing's rate 0.0491 needs **d_seg→~AA-floor (~8e-4) AND d_pose→~3e-4** (√(10·3e-4)=0.055) → 0.086+0.055+0.049≈0.19, **right AT threshold**. Both are OPEN deliverables; no S is guaranteed.

**CONFIRMED:** the config does NOT claim a closed d_pose. `sealed_205` provenance correctly frames w_pose=1.0 + carrier as "pose SLOT shippable-first; a w_pose=0 row does NOT move the pointer" — NO 3.4e-5 claim in the config surface.

**CONCERN C3 (stale-ancestor comment — the exact forgetfulness pattern, in advisory code):** trainer lines 2419-2420 (`realized_verdict`) carry the comment *"deploy pose rides the SOLVED Quantizr stored-pose sidecar, d_pose 3.4e-5"* — an ancestor number cited as if the witness pose were solved. It is a monitoring comment (not a config claim), so non-blocking, but it is the precise pattern the operator flagged. **REVISION R4:** delete/correct it.

## Round 3 — the A/B structure: TWO SEQUENTIAL RUNS, not one-run-dual-close (DETERMINED)

**The optimal A/B is TWO separate training runs, not one d_seg run byte-closed twice** — because the store-nothing carrier CHANGES the training dynamics:
- **table (`sealed_205`):** f0 render = warp of a FIXED real keyframe (`_pc_gt_f0_provider`); the witness INR renders only f1. Pose gradient reaches the witness only through f1 (the PoseNet second frame). Cleaner d_seg attribution.
- **store_nothing:** f0 render = `warp(the witness's OWN plain frame0 render, xi_eff)` (trainer line 1815-1835). Line 1820 verbatim: *"The dxi residual co-grads THROUGH the witness f0 render (the co-adaptation)."* ⇒ the pose loss now ALSO shapes the shared INR decoder via frame0. **Different witness trajectory → cannot be recovered by dual-byte-closing a single run.**

Consequences:
- **The two n600 arms cannot run CONCURRENTLY** (2 × 67.6 = 135 GiB > 128 GiB RAM) → **SEQUENTIAL** (each multi-hour/day).
- **store_nothing's f0→shared-decoder co-adaptation may DEGRADE d_seg** (frame0 is seg-free per `modules.py:108`, but the decoder is shared across frames via per-pair codes). The A/B must WATCH whether store_nothing's pose co-adaptation costs d_seg vs the clean table arm — a genuine confound the run measures.
- **CONFIRMED byte-identity (claim 4):** `store_nothing` = `sealed_205` + exactly ONE flag `--pose-carrier-source generated` (84 vs 83 flags; diff is one line). sealed_205 stays byte-identical to §7 (modulo R2).

## Round 4 — Config coherence + determinism (CONFIRMED)

Emitted launch.sh (measured): `--seed 0` · `--stage-checkpoints` (resumable per-stage) · `--ckpt-every 25` (intra-stage saves) · `--mlx-device gpu` (training gradient; verdict is fp32-numpy CPU authority) · `--ema-decay 0.997` (Quantizr) · curriculum `CE → tau-softplus@300 → muon@726 → l7@1000(==epochs → DEMOTED to ≤1 trailing epoch, the measured-defect fix)` — SANE. Levers: `--render-aa none` + analytic `--lane-render-band` (Wave-D AA correction; brute supersample DISQUALIFIED −49% + decode-over-budget), persistence/topology, island-birth amplification, annealed hosc 1→4 (fixed-β=4 divergence fix), mod-dim 32, adam-beta2 0.999 (== MLX default → byte-identical). **83/83 + 84/84 flags exist in the real argparse (0 invented).** Test suites GREEN: memory-preflight/verdict-chunking/pose-carrier-byte-close **31 passed**; powerplay/gauge/autoconfig **64 passed**.

**Durability note (non-blocking):** first checkpoint is epoch 25 (`--ckpt-every 25`); the first ~25 epochs have no resume point. Acceptable (the OOM cause is fixed + memory-preflight refuses the OOM config), but the run must survive init→train (the exact point the OOM killed it) — watch the first stage boundary.

## Round 5 — Assumption-challenge axis

**Shared assumption the launch operates within:** *"the trained dxi residual WILL close store_nothing's d_pose toward ~3e-4, and the trained witness render IS PoseNet-legible under warp."* — This is **ASSUMED, not evidenced.** It is legitimate that this is OPEN (measuring it IS the run's purpose), but the review must SAY it is open, not assume it closes. The store-nothing arm additionally assumes *"warping the witness's own frame0 render carries pose as well as warping a real keyframe"* — the read-back-through-warp risk the table arm sidesteps. **A negative here (trained d_pose stays catastrophic) re-opens the pose paradigm on the SDF witness** — which is exactly why R1 (a cheap trained smoke) must precede the multi-day n600.

Second assumption: *"mod-dim 32 is d_seg-optimal."* SEALED over the Whitney-floor 19 because 19's d_seg-neutrality is UNMEASURED. Reasonable (d_seg is binding; 32 has headroom + rate slack 0.055<0.081), but it is a SEAL under uncertainty, not a measurement.

---

## REVISIONS (ordered; R1 blocks the full-n600 GO)

1. **R1 [BLOCKING for full n600 — satisfies operator HOLD condition (1)]:** before the multi-day n600 launch, run a SMALL (n24 or n96) **store_nothing TRAINED** pose smoke (`--pose-carrier-source generated`, w_pose>0, train the dxi residual to a stage boundary), byte-close it, and **MEASURE the trained d_pose through the real bit-exact decode.** The per-pair (P,6) residual + the byte-close infra now exist. If the trained d_pose closes toward ~3e-4 (or at least off the catastrophic 4.97 floor), the read-back-through-warp bet holds → full n600 is a PROCEED. If it stays catastrophic, the witness pose is re-opened BEFORE burning the full run. This is measurement-first (CLAUDE.md ANTI-SIGNAL-LOSS rule 3) + the cheapest way to satisfy the operator HOLD.
2. **R2 [defense-in-depth]:** emit `--verdict-batch 32` EXPLICITLY (kills the coupled implicit-default fragility; self-documents the OOM fix in launch.sh; accept the §7 byte-identity break — the §7 argv was the OOM'd one, see C4).
3. **R3 [attribution]:** run the A/B as TWO SEQUENTIAL n600 runs; instrument whether store_nothing's f0→shared-decoder co-adaptation degrades d_seg vs the clean table arm.
4. **R4 [housekeeping]:** delete the stale ancestor "d_pose 3.4e-5" comment at trainer lines 2419-2420 (the forgetfulness pattern in advisory code).
5. **R5 [framing]:** the launch is a MEASUREMENT run. The table arm is rate-doomed (0.51+, attribution/SLOT only); the store_nothing arm sits RIGHT AT the 0.19 threshold contingent on BOTH d_seg→AA-floor AND d_pose→~3e-4 (both OPEN). No S is claimed; only a byte-closed n600 exact row from `upstream/evaluate.py` (CPU/CUDA, NEVER MPS) moves the pointer.

## C4 — the §7 byte-identity tension (for the coordinator/operator)
`sealed_205` is designed byte-identical to the §7 SEALED oracle. R2 (emit `--verdict-batch 32`) breaks that identity. This is CORRECT: the §7 argv is the config that OOM-died; the OOM fix is a NEW required safety flag that belongs IN the launched config. Update the §7 oracle + the triality (DSL/DAG/equations) to the fixed argv, OR keep the implicit default and accept C1 (weaker). Recommend R2 + oracle update.

## Bottom line
The updated config **caught what the last SEAL missed**: the OOM is fixed + self-protected (measured), the pose is honestly OPEN + pre-residual (no borrowed number survives the fail-closed CorrectnessDemonstration), the byte-close is bit-exact for both carriers, and the A/B is correctly two runs. **PROCEED-WITH-REVISIONS: land R1's cheap trained-pose smoke FIRST (it is the operator HOLD condition), then GO the full n600 store_nothing arm as a MEASUREMENT run.** Do NOT rubber-stamp a sub-0.19 claim — none is measured. Pointer 0.19110 UNMOVED.
