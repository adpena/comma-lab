# Re-Audit Re-Founding + MD-Decoupling (2026-06-26)

**Trigger:** operator paranoia after the EMA-shadow-lag catch (78× gap) — "makes me paranoid about many of our negative results... deep math must go deeper than ever" + "research arXiv 2606.25971." A 4-lens systematic re-audit of all load-bearing negatives + a clean achievable-S re-derivation. Pointer UNMOVED contest-CPU **0.19110** (no exact row moved this session — stated plainly).

## THE UNIFIED RE-FOUNDING (all 4 re-auditors converge)

**R1 (axis-confusion):** 6 of 10 load-bearing negatives are axis-suspect → RE-OPEN. The 4 that STAND (realized-through-R, CPU-torch, contest-axis): decode-side floor (Lane A), rate-axis-exhausted-of-the-PR110-archive, prune+KD cliff *(scoped)*, int5 cap *(scoped)*. Highest-impact corrupted: FEED-bj "isotropic witness non-viable 0.037" = lagging-EMA (live 0.0022).

**R2 (capacity-vs-bug):** **0 of 11 capacity-walls are proven fundamental.** 6 likely fixable bugs (collapse / under-train / wrong-LR muon_lr-150× / mis-fit power-law), 2 uncertain-lean-bug, 3 genuine only for a dominated config (prune+KD, int5-retrofit, flat-sidecar). **params⁻⁰·⁷¹ power law is DEAD as a universal law** (its NCA fit came from a collapsed run d_seg 0.508; exponent unstable −0.71/−0.91/−1.52; bc24<bc20 confirms only the sign). The "2×2 capacity-limited" verdict was computed on a diverged bc20@192 arm.

**R3 (harness-fidelity):** 3 of 9 measurement surfaces non-exact (deepmath-smoke + byte-closer-advisory = generator-argmax PROXY, no R/no-SegNet-re-seg; pr95 `apply_eval_roundtrip_nhwc` SWAP live in-tree) + sg_drf EMA-shadow risk. **Highest-impact: the CLAUDE.md "all-class directional −48% decisive lever" ranking rests on the PROXY axis** — never confirmed realized. Trusted/exact: trainer verdict (8cceaa072), verify_e2e, seg_core, contest_score, contest_auth_eval.

**R4 (clean achievable-S, deep-math):** the "rate dead 0.118" is the entropy of the WRONG representation (RGB HNeRV, 94% rendering weights). **Existence proof (measured byte-count): base_ch20 byte-closes at 89,628 B → rate term 0.0597 (half the frontier).** Deepest insight: **seg and pose are the SAME information** — the argmax partition's temporal evolution IS the ego-motion stored as pose (seg=warp); coded jointly (base_seg curves + ego-motion=pose + sparse object residual) the rate and pose terms partially fuse → task-statistic ~15–40 KB (rate 0.01–0.027). **Clean achievable S ≈ 0.10–0.14, binding term = SEG.** Every score-relevant wall except pose's √-flatness is EMPIRICAL. Sub-0.15 corridors: (A) hold base_ch20 rate 0.06 + pose 0.017 → need d_seg < **7.3e-4** (3.6× over current); (B) crush rate to 0.02 → need d_seg < 1.13e-3.

## CONSEQUENCE: the campaign was NOT walled — it was measuring the wrong representation and reading artifacts (EMA-lag, collapse, under-training, proxy-axis, mis-fit power-law) as floors. **Sub-0.15 has clean margin.**

## MEASUREMENT-INTEGRITY FOUNDATION (the durable meta-fix)
- ONE validated realized harness for witness d_seg/d_pose = MLX/numpy render → `_torch_R_to_camera_uint8` → real CPU-torch SegNet/PoseNet (the trainer's verdict path, 8cceaa072). Validate once to ≥6 decimals vs `contest_auth_eval`.
- Report BOTH live AND EMA (EMA decay 0.997 lags fast single-frame descent up to 78×). recompute-S-from-components (never cached final_score).
- Demote all proxy/smoke d_seg (generator-argmax, no R, no SegNet-re-seg) to feasibility-only.
- **NO negative verdict stands until re-checked on this foundation.**

## THE RE-FOUNDED PATH (the decisive pointer-mover)
A from-scratch **task-space / small-basis witness**, trained to convergence with the FIXED recipe, on the validated realized harness, amortized-600, live+EMA, ≥384 render res, targeting **byte-neutral d_seg < 7e-4 at ~0.06 rate → byte-close → contest-CPU exact eval → S < 0.15.**

Math-optimal vehicle ranking (R4): (1) **ego-motion task-space witness** — seg = warp(base_seg, pose) + sparse object residual; pose reused as the seg-warp (free); minimal joint rate; (2) **lever_b class-logit witness** + stored-pose sidecar + directional/step-native basis + chroma; (3) **base_ch20-better-trained** (existence-grounded fastest sub-0.15 row — bank it); retire RGB-detour (PR95/isotropic-RGB) as dominated.

## MD-DECOUPLING (arXiv 2606.25971, Hägele/Hernández-Cano/Kosson/Jaggi, EPFL MLO, Jun 2026) — the optimizer fix for the re-audit's root cause
**Method:** factorize each weight into a fixed-norm direction (hypersphere) + learnable per-row/per-column magnitude gains, updated at separate LRs. **Eliminates weight-decay + warmup; transfers optimal LR across model width without retuning; improves stability; works with Adam AND Muon; scales to MoE.**

**Why it matters here:** R2 found the witness "capacity walls" were optimizer bugs — collapse (destabilizing early updates), wrong-LR/no-LR-transfer (muon_lr-150×), warmup-dependence. MD-Decoupling addresses ALL of these principledly: (a) "destabilizing first-step updates never appear" = anti-collapse; (b) no warmup needed; (c) **LR-transfers across width** → the from-scratch small-basis run needs NO per-bc-size LR retuning (a measured artifact source); (d) Muon-compatible (our optimizer).

**How folded (NO-FAKE):** decisive run #1 uses the KNOWN-GOOD validated recipe (8cceaa072 stabilizers + muon_lr fix + EMA-warmup). **MD-Decoupling is the next-iteration optimizer ablation** — does it converge lower/faster, transfer LR across bc-size, remove warmup, and reduce the EMA-lag-prone fast-descent? A candidate MEASURED win, not assumed. OSS: not yet released (re-implement from the paper; ~per-row/col gain reparam + separate-LR + hypersphere projection).

Cross: FEED-bj/be/bd/bc/bk/bl in `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`; the 4 re-auditor task outputs (a6a0a8e2 / a5efd5f5 / ae626ffc / a3ab7a85); the witness fix-pass 8cceaa072.
