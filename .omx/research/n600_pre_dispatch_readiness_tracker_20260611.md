# n600 pre-dispatch readiness tracker — the gate checklist before the $100 spend (2026-06-11)

**Status:** $100 APPROVED (operator), but **nothing deploys until every gate below clears + the adversarial
symposium returns PROCEED.** Frontier UNMOVED 0.19109982 [contest-CPU], 177,169 B. Operator binding additions
2026-06-11: (i) significant time/resources authorized for LOCAL MLX training groundwork (P1 favored);
(ii) the vendored PR95 TORCH trainer (P2) must ALSO be adversarially reviewed; (iii) whichever vehicle,
the run MUST be pausable+resumable, best-checkpoint-tracked, with FULL telemetry + tooling, properly
configured BEFORE spend.

## The pointer-mover thesis (quantified, config agent a97c23f)

Frontier is RATE-DOMINATED (rate 0.118 = 62% of S). Measured byte budgets (stored_latent int8, n600):
**base_ch=20 tie=2 = 86,660 B (HALF the frontier) → S≈0.1309 (SUB-0.15) IF the basin holds.** PR95 proved
the 5.6e-4 basin at base_ch=36 (246KB, bigger than frontier, no rate win). **The bet: the SAME basin at 30%
of the params.** Recipe: FAITHFUL PR95 (AdamW 1-7 + Muon 8) is default (muon-throughout only 1.38× at n8 —
not decisive). Ladder: primary base_ch=20 tie=2, fallback base_ch=24 tie=2.

## The vehicle decision (operator + symposium)

The MLX capstone is **Apple-only — cannot run on Modal NVIDIA.**
- **P1 = local-MLX capstone** — free, slow, the operator's dream; significant investment AUTHORIZED for the
  groundwork. Feasibility + throughput unblock + resumability: agent acf1665a (in flight).
- **P2 = vendored PR95 torch trainer on Modal** — $100, fast, basin-proven; run at base_ch=20 for the rate win.
Decision pending acf1665a's MLX feasibility verdict; operator leans P1.

## The gate checklist

| # | Gate | Owner | Status |
|---|---|---|---|
| 1 | Config OPTIMAL + NON-ARBITRARY + bug-swept | a97c23f | ✅ DONE — bug-free (BUG-A fixed; BUG-B not a bug for faithful path; all else clean-faithful, decoder bit-exact). 1 knob to pin: `--ema-decay 0.999`. |
| 2 | Recipe-bug-lens findings re-audit | ae07ebd | ✅ DONE — TIER-1 capacity walls FALSIFIED-reactivate; Cool-Chic/B1 PROVISIONAL; rate/CUDA-neg HOLD. |
| 3 | Corpus-wide negative-results resurrection | ac792687 | ✅ DONE — surfaced R1 (cheaper lever-C path) + R2 (unfixed torch-EMA). |
| 4 | R1 (cheaper lever-C un-falsify?) + R2 (fix torch-EMA warmup #86) | aec9b73b | ✅ DONE — R1 does NOT obviate (lever-C "FALSIFIED" was partly EMA-artifact — d_seg moves to 0.414 — but still ~740× above frontier, real seg antagonism; **n600 remains the path**). R2 = real fleet-wide torch-EMA-warmup bug FIXED + STRICT gate Catalog #388 (5 instances fixed). Highest-EV fleet re-test: B1 "clean PR95 mean-fields 0.505" @33% epochs under warmup-EMA. |
| 5 | #15 advisory↔exact custody (float-render vs int8-bicubic d_seg gap) | aab68dbe | ✅ **PASSES** — gap d_seg 2.5e-5 / d_pose 5.6e-4 = 0.013% of descent; exact int8 path is LOWER (advisory is conservative, not optimistic). Export/inflate fully wired (82.5KB int8 archive → numpy inflate). The n600 optimizes a number the archive honors. |
| 6 | Local-MLX feasibility + throughput + resumable checkpointing | acf1665a | ✅ DONE — **FEASIBLE as a multi-week resumable run; durable $0 daemon LAUNCHED + measuring the n600 basin depth.** Fast-approx-gradient descent-equivalent (safe) but scorer-backward is >97% of step on both backends (no throughput unblock). torch-CPU 19.4 vs MLX-GPU 26.6 s/step (MLX slower locally — FP32-exact override). |
| 7 | **Adversarial review of BOTH vehicles** (MLX capstone + vendored PR95 torch) | a431c347 + config/recipe agents | ✅ DONE — torch vehicle 3-clean-pass (base_ch=20 sound, objective line-identical, parse-back-fidelity bug fixed); MLX capstone via config + vendored-port diff. |
| 8 | **Resumability + best-checkpoint + FULL telemetry + tooling**, kill+restart-verified | acf1665a (MLX) + a431c347 (torch) | ✅ DONE BOTH — kill-9 bit-identical resume; durable JSONL+dashboard telemetry; best-by-canonical-score. |
| 9 | 10-min step-time SMOKE → measured GPU-hour / wall-clock at n600 (epoch-budget-vs-basin risk) | queued (pre-commit) | ⏳ QUEUED — $100 buys ~12-16k compressed epochs vs PR95's 29,650. |
| 10 | Per-substrate adversarial grand-council SYMPOSIUM (cargo-cult #303 + 9-dim #294 + observability #305 + recursive 3-clean-pass) → PROCEED | symposium | ⏳ FIRES LAST, consuming 1-9. |

## Binding requirements (operator, must hold before spend)

- **Resumability**: pause→resume from BEST checkpoint with ≤1-checkpoint loss on death (the 2×2 ablation was
  lost to exactly this gap). Verified by a real kill+restart test, not asserted.
- **Best-checkpoint tracking**: the exported/eval'd checkpoint is the best-by-exact-d_seg (canonical chain),
  not the last — with EMA-shadow warmup correct (the R2 fix).
- **Full telemetry + tooling**: per-stage/epoch d_seg/d_pose/rate trajectory (canonical chain), loss curves,
  LR/clip/EMA traces, byte-budget, durable JSONL + dashboard, marker-on-exit — for BOTH vehicles.
- **Both vehicles adversarially reviewed** before either is chosen.

## Bottom line

The config is clean and the path to sub-0.15 is quantified (base_ch=20 basin → S≈0.13). What gates the spend
is NOT the config — it's (a) could R1 obviate it with a cheaper carrier, (b) does the advisory number transfer
to the archive (#15), (c) the vehicle/resumability/telemetry readiness the operator requires, and (d) the
symposium PROCEED. Frontier UNMOVED; the rigor is the point on a big step.
