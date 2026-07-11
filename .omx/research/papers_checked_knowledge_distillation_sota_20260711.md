# Papers checked — knowledge-distillation SOTA survey (2026-07-11) — anti-re-research ledger

Source survey: `.omx/research/distillation_sota_survey_20260711T120058Z.md`. Verdicts: ADOPT-CANDIDATE (feeds a named next
action) / WATCH / CONFIRM-not-lever (ratifies something we already do) / NOT-LEVER (checked, no path for us). All
SPECULATIVE-for-us until organ-backtest / n600 row. Consumer: #426 costate organ + witness KD.

| arXiv / ref | Paper | Verdict | One-line |
|---|---|---|---|
| 2002.08676 | Berthet et al., Differentiable Perturbed Optimizers | **ADOPT-CANDIDATE #1** | smoothed-argmax λ through the REAL frozen SegNet; ε ≡ our τ=ε=ħ knob |
| 1912.02175 | Vlastelica et al., Blackbox Combinatorial Solvers | ADOPT-CANDIDATE (with #1) | linear-interp gradients through discrete solvers; sibling of above |
| 1611.01144 / 1611.00712 | Gumbel-softmax / Concrete | CONFIRM-not-lever | the analytic temperature special case; already implicit in margin surrogate |
| 1705.08790 | Lovász-Softmax (Berman) | WATCH | convex IoU surrogate; our d_seg is 0-1 per-pixel so margin is tighter; use if IoU-like term appears |
| 1803.00443 | Srinivas & Fleuret, Jacobian Matching | **ADOPT-CANDIDATE #2** | Jacobian-KD ≡ noise-KD; THE λ-fidelity training objective |
| 1706.04859 | Czarnecki et al., Sobolev Training | ADOPT-CANDIDATE (with #2) | random-projection Jᵀv matching = tractable dense-Jacobian form |
| 2301.12006 | Backward-pass knowledge in KD | WATCH | 2023-24 follow-on to Jacobian matching |
| 1905.10108 | Learning Surrogate Losses | WATCH | learn a differentiable proxy of a non-diff metric |
| 2202.13197 | Relational Surrogate Loss Learning (ICLR22) | WATCH | fit surrogate to preserve RANKINGS of S — all a duty-queue needs |
| 2604.25530 | Canonical KD surprisingly effective for segmentation (2026) | **ADOPT-CANDIDATE #3** | at matched compute, plain logit+feature KD beats CWD/CIRKD/BPKD; reframes seg-KD defaults |
| 2011.13256 | CWD channel-wise seg KD (ICCV21) | CONFIRM-not-lever | classic seg-KD baseline; dethroned at matched compute |
| 2204.06986 | CIRKD cross-image relational (CVPR22) | CONFIRM-not-lever | ditto |
| 2306.08075 | BPKD boundary-privileged KD (WACV24) | WATCH | edge/body loss split; its SAMPLING idea survives (our annulus L66), its loss doesn't |
| ICCV25 ACAM-KD | adaptive cooperative attention masking | NOT-LEVER | zoo member, compute-confounded |
| 1903.04197 | Structured KD for semantic segmentation (CVPR19) | CONFIRM-not-lever | seminal seg-KD anchor |
| 2503.13053 | Uncertainty-aware 6DoF pose KD | WATCH | keypoint-uncertainty-weighted transfer; PoseNet head already smooth |
| 2503.14097 | SCJD 3D-pose sparse-correlation KD (ICME25) | NOT-LEVER | 3D-HPE specific |
| 2404.03518 | SDPose circulation self-distillation | NOT-LEVER | tokenized-pose specific |
| 1805.05532 | Heo BSS adversarial boundary-supporting samples | **ADOPT-CANDIDATE #4** | train surrogate ON teacher-boundary samples; margin field = free sampler |
| 2108.07969 | RSLAD robust soft labels | WATCH | robust-teacher soft labels; boundary-transfer lineage |
| AdaAD (2312.05508-line) | adaptive adversarial distillation | WATCH | inner-max aligned to teacher-student discrepancy |
| 2306.04431 | Faithful KD | **ADOPT-CANDIDATE (acceptance test)** | ball-agreement = the right "captured the separatrix" criterion |
| 2605.21999 | Why robust teachers fail (2026) | WATCH | failure taxonomy for robustness transfer |
| 2506.07666 / 2409.01627 / 2402.15586 | ProARD / DGAD / heterogeneous robust teachers | NOT-LEVER (for now) | robust-KD zoo |
| 1803.09043 / 1906.00697 | ADV-EMB + game-theoretic adversarial embedding | CONFIRM-not-lever | steganography attacks CNN detectors via surrogate gradients = our inverse-steg frame, ratified |
| 2306.13649 | GKD on-policy distillation (ICLR24) | **ADOPT-CANDIDATE #5** | reverse-KL on student-generated inputs; correctness condition for in-loop surrogates |
| 2604.00626 | Survey of on-policy distillation (2026) | CONFIRM-not-lever | maps the field; industrial default |
| 2203.08679 | DKD decoupled KD (CVPR22) | CONFIRM-not-lever | non-target-class mass carries boundary info = why soft labels work |
| 2403.01427 | Logit standardization KD (CVPR24) | ADOPT-CANDIDATE (hygiene) | z-score logits pre-KD; free plug-in |
| 2402.11148 | Transformed Teacher Matching | WATCH | temperature-asymmetric = Rényi-regularized matching |
| 2305.15712 | DiffKD diffusion-denoised features | NOT-LEVER | heavy machinery, no fit |
| 1904.05068 | RKD relational KD | WATCH | distance/angle relations = subspace-geometry distillation shape (BSF) |
| 1910.10699 / 1412.6550 / 1612.03928 | CRD / FitNets / AT | CONFIRM-not-lever | seminal feature-KD anchors |
| 2510.12615 | Functional perspective on KD (2025) | WATCH | theory of what transfers |
| 2508.10104 | DINOv3 (Meta 2025) | **WATCH-high (Gram anchoring)** | Gram-anchor dense features to early ckpt vs long-run drift — same disease as our 10k+ epoch runs |
| 2304.07193 / 2111.07832 | DINOv2 / iBOT | CONFIRM-not-lever | SSL-distillation anchors |
| 1805.04770 | Born-again networks | WATCH | $0 witness experiment shape: retrain vs own EMA soft output |
| 2002.05715 | Mobahi self-distillation theory | CONFIRM-not-lever | few rounds regularize, many collapse |
| 2306.13092 | SRe2L dataset distillation | NOT-LEVER | we have the real video + free teacher queries |
| 2604.18811 / 2606.18209 | Hard truths about soft labels / distilled-vs-coreset (2026) | CONFIRM-not-lever | DD knowledge lives in soft RELABELING, not pixels — spend there if ever needed |
| 2505.13300 | DD-Ranking | NOT-LEVER | DD evaluation hygiene |
| 2502.05673 | DD evolution survey | NOT-LEVER | map only |
| 2311.18828 / 2405.14867 | DMD / DMD2 | CONFIRM-not-lever | distribution-matching via two score fields; idea-transfer only |
| 2410.11081 / 2510.08431 | sCM / rCM | WATCH | continuous-time consistency at scale; no direct path |
| 2202.00512 / 2303.01469 / 2209.14988 | progressive distillation / CM / SDS | CONFIRM-not-lever | seminal generative-distillation anchors |
| 2502.08606 | Distillation Scaling Laws (Apple, ICLR25) | **ADOPT-CANDIDATE (design law)** | frozen-free-teacher = the regime where distillation dominates; sizes any surrogate |
| 2311.07052 | Law of capacity gap | WATCH | optimal teacher ~linear in student scale |
| 2106.05237 | Beyer et al., Patient & Consistent (function matching) | ADOPT-CANDIDATE (recipe) | same-views + long training beats clever objectives |
| 2505.15442 | Generalization-vs-fidelity paradox | WATCH | agreement ≠ the objective |
| 2504.00870 | DiffDFKD data-free KD w/ diffusion (2025) | WATCH | query-synthesis machinery if boundary sampling needs a generator |
| 2507.04119 | DFKD non-transferable-teacher OOD trap | NOT-LEVER | niche |
| OSS: torchdistill / mdistiller / MMRazor / CIRKD-repo / trl.GKDTrainer | frameworks | REFERENCE-only | reference impls; we build MLX-first repo-native |
