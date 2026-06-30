# Post-Muon application plan — apply all (as appropriate + optimal) to the current run

**UTC** 2026-06-30T17:10Z · **tag** `[macOS-MLX advisory · research/design · NON-PROMOTABLE]` · **pointer 0.19110 UNMOVED.**
Operator directives 2026-06-30: "work toward applying all as appropriate and optimal to current run" + "dseg keeps going down but it's very slow." Synthesizes the 4-thread deep-math (RL parked / inverse-steg surgical / causal screw-warp / factorization-observability / topology-birth-death / thermodynamics) onto the LIVE Muon arm. means≠ends: this is the application SPEC + a ready-to-fire recipe; the only END is a byte-closed exact row < 0.19110. NO autonomous heavy-GPU launch (one GPU, Muon owns it; await operator steer).

## 0. The "very slow" verdict (load-bearing)
The Muon tail (0.003988@ep900 → 0.003805@ep950 → 0.003718@ep975, decelerating) is **critical slowing down near a rate-distortion topological transition** — Agmon–Benger–Ordentlich–Tishby ISIT 2021 (arXiv:2103.02646), proven for the Arimoto–Blahut class our annealing curriculum descends; our curriculum IS deterministic annealing (Rose 1998, F=D−T·H, τ=temperature). **Consequence:** riding Muon longer = diminishing returns on the power-law tail; the EXPONENT is broken by the levers (directional basis −48%, screw-warp, root-tracking schedule), not by more epochs. This TEMPERS any "Muon plateaued, kill it" — it's a critical point, not a dead end (terminal-conclusion-cross-check discipline).

## 1. Division of labor (MEASURED, the appropriateness map)
- **Lane = PRIMED, the witness's job (trained/stored residual).** Attribution: 47.2%→24.3% mislabeled, margins moving toward GT every transition, Muon converting it. Screw-warp through-R: Lane ~39% flip, WARP-UNEXPLAINABLE → must be learned/stored. north-star: pose-R² 0.363 (low). → KEEP TRAINING + store-the-flip last.
- **Road = STUCK but CAUSAL (the screw-warp's job, ~0 byte).** Attribution: error mass GROWING 21.5%→38.8% (now largest), margins flat/negative (witness antagonizes it). north-star: pose-R² 0.527. screw-warp through-R: warp HELPS Road −8% (0.0155→0.0142), fixes sky/hood by construction. → the witness should STOP fighting Road per-pixel; the se(3) screw / lane-prior-phi1 geometric guard handles it.
- **The annulus is the whole game** (97.7–98.5% of error codim-1; boundary anisotropy 9.56:1 gradient / 37.8:1 structure-tensor; separatrix AUC 0.999). 1D ridges + saddles (triple junctions n≈400–834/frame; dashed-lane = birth-death pairs, ~2700 events/600 frames).

## 2. The application — almost all levers are ALREADY trainer flags (config, not rebuild)
| Lever (synthesis) | Trainer flag(s) | Routed to |
|---|---|---|
| Directional basis (−48% exponent-change) | `--self-orient --n-dir-freqs 2 --freq-across 32 --freq-along 4 --reorient-every 50` | annulus (already ON in Muon) |
| Lane priming (PRIMED) | `--margin-saliency-weight/-target/-tau/-start-epoch` + `--hardness-weighted/-band/-power/-source/-oversample` | Lane annulus (moving margins) |
| Thin-Lane STUCK orbit | `--lane-thin-weight/-radius 2/-target/-class 1/-start-epoch` | the ~2px lane tube |
| UNIWARD on STUCK texture (inverse-steg) | `--margin-saliency-uniward --margin-saliency-uniward-beta` | thin-lane sub-core |
| Road geometric guard (causal, free) | `--lane-prior-phi1 --lane-prior-phi1-mode/-bias-scale/-dash-gate/-source-pair` | Road↔Lane separatrix (openpilot deg-3) |
| Conditioning/capacity | `--code-spectral-entropy-weight --film-stiefel` (DM1, default-off; firewall-gated) | code spectrum |
| Bandwidth anneal (graduated-non-convexity) | `--max-bank-freq 16→32→64` | critical-slowing cure |

**OPEN (need a real build, not a flag):** (a) the se(3) screw-warp v2 vehicle (Road handled deterministically; the bigger build — defer to optimal-form post-agents); (b) the root-tracking anneal scheduler (`--optimizer custom`, designed `per_stage_fractal_optimizer_priming_reheat_anneal_design`); (c) arch resize to the solved RD-optimum (mod-dim 32→21, hidden 96→120) — a FROM-SCRATCH change (breaks warm-start ckpt shape), so it belongs to v2, not the warm-start.

## 3. The ready-to-fire warm-start re-treatment (HOLD for steer + Muon-done; one GPU)
A NEW stage warm-started from the Muon BEST (the current run's preserved ckpt — whole-run archived), applying the loss-flag levers (arch held = warm-start-compatible). Per-lever OPTIMAL VALUES come from the θ* per-lever A/B (#183) — do NOT invent values; this is the STRUCTURE:
```
.venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py \
  --resume-from experiments/results/levelset_thetastar_muon_arm/levelset_witness_ema_BEST.npz \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz --num-pairs 200 \
  --mlx-device gpu --async-verdict \
  --self-orient --n-dir-freqs 2 --freq-across 32 --freq-along 4 --reorient-every 50 \
  --margin-saliency-weight <θ*> --margin-saliency-target lane --margin-saliency-tau <θ*> --margin-saliency-start-epoch 0 \
  --hardness-weighted --hardness-source margin --hardness-band <θ*> --hardness-power <θ*> --hardness-oversample <θ*> \
  --lane-thin-weight <θ*> --lane-thin-radius 2 --lane-thin-target <θ*> --lane-thin-class 1 --lane-thin-start-epoch 0 \
  --lane-prior-phi1 --lane-prior-phi1-mode <θ*> --lane-prior-phi1-bias-scale <θ*> --lane-prior-phi1-dash-gate \
  --margin-saliency-uniward --margin-saliency-uniward-beta <θ*> \
  --max-bank-freq 64 \
  --out-dir experiments/results/levelset_retreat_warmstart_<utc>
```
Resumable + per-stage ckpts + EMA-shadow + best-preservation are already hardened (3da9a6b10) + the whole-run archiver runs. The fire TRIGGER = Muon's marginal Δd_seg/epoch falls below the lever's expected gain (a plateau detector on `levelset_best.json`) OR operator steer.

## 4. The v2 (from-scratch optimal-form, after the 2 running agents land)
The screw-warp causal vehicle + arch at the solved RD-optimum (mod 21 / hidden 120) + root-tracking anneal + REHEAT (anti-collapse noise on collapsing eigendirections) + the factorized 4D observability (se(3) screw temporal factor × per-class depth × K-plane spatial × Morse-Smale persistence) — gated on the factorized-4D + birth-death agents (running) for optimal form. This is the RATE-half move (smaller representation = the binding sub-0.15 lever).

## 5. Decision points (operator)
1. **Fire the warm-start re-treatment now-ish** (when Muon's marginal gain < lever EV) — it's the same GPU, sequential after Muon; needs your go (no autonomous heavy launch).
2. **Run the θ* per-lever A/B (#183)** to fill the `<θ*>` values at optimal form before the compose stage (else use prior-measured defaults).
3. **Root-tracking scheduler** = the wall-clock-optimality build (same d_seg, fewer epochs) — worth building before the n600 burn.

Anchors: thermo brief (Rose 1998 / Agmon-Tishby 2103.02646 / Chan-Vese / ZOGY / DRUID), ours-code map, `screw_warp_through_R_gap2_20260629`, `witness_per_stage_attribution_20260630`, `north_star_task_geometry_causal_factors_20260629`. Pointer 0.19110 UNMOVED.
