# Findings re-audit through the recipe-bug lens — the verdict-chain re-classification (2026-06-11)

**Authority:** $0 read-only classification (agent `ae07ebd`, `findings_reaudit_recipebug_20260611`), persisted by
parent (subagent harness blocked its own file write). NO MPS, no dispatch. Frontier UNMOVED 0.19109982
[contest-CPU], 177,169 B. Catalog #307: "relied on the buggy loop" ⇒ IMPLEMENTATION-FALSIFIED, PARADIGM
INTACT ⇒ REACTIVATE-on-fixed-recipe (NOT "old conclusion was wrong" without re-measurement). Trigger:
operator "review all findings and results and interpretations and candidates and vehicles in light of this
new" + the recipe-bug discovery (`configure_stage` dropped muon_lr 0.03→2e-4 + 100%-clip; fixed a9888191c).

## The harness map (who touched the bug)

| harness | BUG-A? | epochs measured | verdicts it produced |
|---|---|---|---|
| `CapstoneTrainer`/`configure_stage` (muon_throughout) | **YES — the bug** | n8–48, ≤120ep | capstone "capacity wall / muon asymptote 0.0025", c1prime "stuck 0.010", 2×2 ablation, "CE plateaus 0.008 / 85K fights physics" |
| `ScoreAwareTrainer` (Cool-Chic) | NO BUG-A (AdamW 3e-3); shares BUG-B clip=1.0+cosine | **120ep only** | Cool-Chic "walls 0.014", smaller-basis-by-rate REFUTED |
| B1 faithful `mlx_pr95_port` 8-stage | NO (AdamW 1-7) | **ep1000/3000 = 33%** | "clean PR95 also mean-fields d_seg≈0.505" |
| NeRV/HiNeRV/SNeRV fleet | NO — own loops, receiver-survival | N/A | HiNeRV/SNeRV "wall" = parseback/inflate survival |
| Frozen-frontier byte work | **no training loop** | N/A | rate entropy-floor, selector exhausted, lossy coarsening |

## TIER 1 — FALSIFIED-REACTIVATE (touched the buggy muon path). Highest EV.

1. **Capstone base_ch=20 "capacity wall / muon asymptotes ~0.0025"** — YES (dropped 2e-4 + clip 1.0; the
   "asymptote" is partly the BUG-B LR floor). **P0** = #90 reframed; fixed A/B descends 42× where this froze.
2. **"Every $0 door is a retrain — the retrain would ALSO wall / #90 is a FAKE spend"** — YES, the "retrain
   walls" premise IS the throttled plateau. **P0.** The "fake spend" verdict is itself falsified; spend justified.
3. **c1prime "curriculum plateau ~0.010" + BUG-C "smooth raises d_seg"** — YES. **P1.** Fixed A/B already shows
   stage-3 smooth REFINES (0.0165→0.0120). Re-run n48/n600 to bank.
4. **2×2 capacity ablation read as a capacity-vs-pairs curve** — YES (every arm throttled). **P1.** It measured
   the throttle, not capacity. Run the unrun bc24 + an UNTIED arm on the fixed recipe before any capacity slope.
5. **"CE plateaus ~0.008 cannot cross to basin" + "85K fights param↔d_seg physics"** — PARTIAL→mostly YES; the
   param-curve rests on a 2026-05-09 council *projection, not a measured curve*. **P1.** Reactivate plateau, reopen curve.

**n600 fold-in (re-audit recommendation):** ONE campaign closing #1–#5 — corrected full 8-stage curriculum
(BUG-A fix routed through `configure_stage`), at **frontier-CLASS capacity (base_ch≈22–24, ~100–160K, NOT the
85K tied basis), with an UNTIED arm**, n600 paid GPU. NOTE the standing tension: frontier-class capacity ≈
frontier bytes ⇒ the rate win then requires the run to BEAT the frontier's d_seg/d_pose at ~equal bytes (a
distortion win), OR a separate smaller-basis arm must hold the basin for a rate win. The config agent +
symposium must resolve this explicitly. **Gate on TIER-3 #15 first.**

## TIER 2 — PARTIAL-VERIFY → PROVISIONAL (different harness, but under-epoch / sister-throttle confound)

6. **Cool-Chic "walls 0.014" / smaller-basis-by-rate REFUTED** — NO BUG-A, but only **120 epochs** + shares the
   grad_clip=1.0+cosine family. **DOWNGRADE to PROVISIONAL; do NOT cite it to refuse the n600.** Re-run L2/L3 at a
   large epoch budget. **P2.**
7. **B1 "clean PR95 mean-fields ≈0.505"** — NO BUG-A, but read at **33% of epochs**. **P2.** Not a settled negative;
   let it reach ep3000+ and read live-render exact d_seg.
8. **#40 "skip-free decoder → mean-field → 0.5"** — arch finding entangled with under-training. **P2.** Wire F1
   bilinear-skip, isolate at sufficient epochs.
9. **A5 AFSR-1 "KILLED-AT-IMPLEMENTATION"** — kill of *continuation* holds; its "train-from-INIT fresh basin"
   reactivation is now MORE attractive (a fresh small basis descends when not throttled). **P2.** Fold into the
   n600 capacity arm.

## TIER 3 — STILL-HOLDS (no training loop / independent paradigm). DO NOT reactivate (over-reactivation = signal loss).

10. Rate entropy-floor (selector 593/600 argmin exhausted, 7.999 bits/byte, E1 frozen-bytes) — frozen archive, holds.
11. Frontier decoder-axis waterfill + the GT-decode bug-class fix (yuv420_to_rgb vs rgb24) — frozen-frontier surgery, holds; the live highest-EV byte path.
12. Lossy coarsening 0.3517 [contest-CUDA negative]; decoder-axis c1/c2/c3; QAT-recovery — CUDA-confirmed, no recipe dependency, holds; DO NOT redispatch.
13. HiNeRV/SNeRV receiver/parseback/inflate-survival walls — witness-compiler paradigm, own loops, holds.
14. d_pose "solved" via stored 6-dim GT pose + FiLM — stored-answer mechanism, recipe-irrelevant; holds (int8/600-pair survival still to verify).
15. **THE ADVISORY↔EXACT DECOUPLE — STILL-HOLDS, the BINDING GATE that survives the recipe fix.** Capstone
    d_seg/d_pose are LIVE FLOAT MLX-render numbers, NOT the int8 archive / bicubic-inflate the scorer sees. **No
    capstone advisory number is a trustworthy `inflate.sh→evaluate.py` predictor until a $0 reloaded-int8 +
    bicubic-inflate smoke runs. CLOSE THIS BEFORE THE n600 SPEND** — a long train optimizing a float the int8
    archive doesn't honor wastes the spend even with the recipe fixed. **This MUST be closed by a $0
    reloaded-int8 + bicubic-inflate smoke BEFORE dispatch — it falls in the config agent's export→byte-close
    →inflate-parity remit and is an EXPLICIT symposium gate; NOT yet run as of this writing.**

## Bottom line

- **Survives:** every TIER-3 verdict (rate floor, frozen-frontier byte exhaustion, lossy-coarsening CUDA-negative,
  HiNeRV/SNeRV receiver walls, the advisory↔exact custody gate). Reactivating these would be the over-reactivation error.
- **Must be re-run:** the whole capstone "capacity wall" cluster (TIER 1, implementation-falsified, reactivates on the
  fixed recipe); Cool-Chic REFUTED + B1 mean-field downgrade to PROVISIONAL (under-epoch/sister-throttle confounds).
  The "#90 fake spend" verdict is itself falsified — the spend is justified.
- **Single highest-EV action:** fold TIER-1 #1–#5 (+ #9 "fresh basin") into ONE de-risked paid n600 — corrected
  8-stage curriculum, frontier-class capacity, untied arm — **after** closing the TIER-3 #15 advisory↔exact custody
  gate ($0 reloaded-int8 + bicubic smoke).
