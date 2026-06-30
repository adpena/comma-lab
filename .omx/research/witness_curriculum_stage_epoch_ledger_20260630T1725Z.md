# Witness curriculum stage/epoch/d_seg LEDGER — the recall for fine-tuning + n600

**UTC** 2026-06-30T17:25Z · `[macOS-MLX advisory · realized/deploy d_seg · NON-PROMOTABLE]` · **pointer 0.19110 UNMOVED.**
Operator 2026-06-30: "recall as we fine tune what we ran for how many epochs at which stages; fine tune the score down as much as possible, then from-scratch n600, then fine tune the schedule/curriculum itself." This is the deterministic-reproducibility recall, assembled from the preserved per-stage ckpts + configs + attribution + ckpt-archive manifest.

## The run lineage (openpilot-seeded n200, curriculum CE→Tau→L7→Muon, warm-start chain)
Stage-start config (from ckpt `__cfg`): tau_softplus@300, l7@600. Per-stage ckpts PRESERVED (resumable).

| Stage | Epochs | # ep | realized d_seg (96-pair / n200 deploy) | ckpt | notes |
|---|---|---|---|---|---|
| CE            | 0–299   | 300        | 0.005443 (end) | `levelset_openpilot_seeded_n200_DEPLOY/levelset_ckpt_stageCE_ep299.npz`  | static core (Road/Undriv/MyCar ~solved); Lane 47.2% mislabeled; 98.5% of error on annulus |
| Tau_softplus  | 300–599 | 300        | 0.004563 (end) | `…/levelset_ckpt_stageTau_ep599.npz` | **biggest single drop −0.00088**; primes 38.4k px (margin+0.366); Road drift BEGINS (regresses 15.5k px) |
| L7            | 600–725 | 126        | ~0.004227 (knee ~ep700; arm deploy ~0.004287) | `levelset_thetastar_l7_arm/levelset_ema_stageL7arm_final_ep725.npz` | modest refinement; knee fast |
| Muon          | 726–975+| 250+ (LIVE)| 0.003988@900 → 0.003917@925 → 0.003805@950 → **0.003718@975** | `levelset_thetastar_muon_arm/levelset_witness_ema_BEST.npz` (+ MuonStart_ep726; l7→MuonStart +0.000024 = temp-reset artifact, NOT learning) | the d_seg finisher; CRITICAL-SLOWING tail (Agmon-Tishby ISIT'21), still descending |

Total run so far ≈ **975 epochs**. dense best-archive: ep900/925/950/975 (whole-run archiver keeping every best + latest).

## Curriculum-tuning signal (feeds step 4 + the n600 schedule)
- **CE+Tau (600 ep) did the bulk early drop** (→0.004563); the static core + the prime.
- **L7 short knee** (~ep700, 126 ep, modest) — over-long L7 is wasted (DOE: stages 5-7 were rate-tuning, Muon is the finisher).
- **Muon = the long slow finisher** (250+ ep, 0.004227→0.003718, decelerating = critical slowing). The thermo cure = ROOT-TRACKING anneal (slow through critical-τ, fast between → same d_seg, fewer epochs) — the wall-clock-optimality lever for the n600 burn.
- Per-stage softmax_temp: CE 0.529 → Tau 0.050 → L7 0.136 → Muon-reset 1.0 → 0.216 (attribution).

## The 4-step plan (operator)
1. **RECALL** (this ledger) ✓.
2. **Fine-tune score down on the current lineage** — the warm-start re-treatment (Lane priming + Road geometric guard + UNIWARD + directional; `post_muon_application_plan_optimal_form_20260630T1710Z.md`), held for steer (one GPU, Muon owns it).
3. **From-scratch n600** — the full-scale run with the tuned schedule.
4. **Fine-tune the schedule/curriculum itself** — informed by this ledger (generous CE+Tau, short L7, root-tracked Muon) + the thermo root-tracking + the θ* per-lever A/B (#183).

Anchors: `witness_per_stage_attribution_20260630`, `post_muon_application_plan_optimal_form_20260630T1710Z`, `pr95_dseg_30k_convergence_deepmath`, the thermo brief (root-tracking arXiv:2306.09790). Pointer 0.19110 UNMOVED.
