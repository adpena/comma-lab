---
name: OWV3 WAVE-3 REFINEMENT PLAN — 6 chain-eval candidates ready for dispatch (post R7 1.013 win)
description: 2026-05-01 ~13:10 UTC. Wave-3 sweep built 121 candidates locally at experiments/results/lane_g_v3_owv3_wave3_refinement_20260501/archives/. 6 selected for chain eval (2 conservative + 2 mid + 2 aggressive). Predicted band [0.998, 1.012]. Dispatch action: launch fresh Vast.ai 4090 ($0.25-0.30/hr), SCP 6 archives (3.6MB total), chain-eval via remote_archive_only_eval.sh. Total cost ~$1, total wall-clock ~3h.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## Wave-3 selection (built locally 2026-05-01 ~13:10Z)

Sweep params: bbr ∈ {0.66, 0.665, 0.67, 0.675, 0.68, 0.685, 0.69, 0.695, 0.70, 0.71, 0.72} × protect ∈ {0.0013-0.002} × aggr {1e-05}. 121 candidates, all byte-feasible vs PFP16 frontier 686,635 bytes.

R7 baseline: owv3_0001_bbr0p7_protect0p0013, **631,473 bytes, score 1.0134** [contest-CUDA RTX 4090].

The 6 chain-eval candidates:

| Label | Candidate | bbr | protect | bytes | Δ vs R7 | Predicted score* |
|---|---|---|---|---|---|---|
| cons | owv3_0043_bbr0p695_protect0p002 | 0.695 | 0.002 | 624,419 | -7,054 | ~1.009 |
| cons | owv3_0032_bbr0p7_protect0p002 | 0.700 | 0.002 | 624,996 | -6,477 | ~1.009 |
| mid  | owv3_0076_bbr0p68_protect0p002 | 0.680 | 0.002 | 621,914 | -9,559 | ~1.007 |
| mid  | owv3_0065_bbr0p685_protect0p002 | 0.685 | 0.002 | 622,407 | -9,066 | ~1.008 |
| aggr | owv3_0120_bbr0p66_protect0p002 | 0.660 | 0.002 | 617,410 | -14,063 | ~1.005 |
| aggr | owv3_0119_bbr0p66_protect0p0018 | 0.660 | 0.0018 | 618,443 | -13,030 | ~1.005 |

*Predicted score assumes PoseNet holds at baseline-similar levels. If PoseNet drifts by 5% (like at bbr=0.65 → 0.00473), aggressive candidates regress to ~1.013-1.018.

The "protect=0.002" column is HIGHER than R7's 0.0013 — that means MORE channels are protected, compensating for the more aggressive bit budget. This is the Pareto-trick: trade SegNet headroom (already at saturation 0.00401) for more aggressive rate compression on PoseNet-irrelevant channels.

## Dispatch plan

**Step 1: Launch Vast.ai 4090 instance** (~5-10 min boot)
```bash
.venv/bin/python scripts/launch_lane_on_vastai.py phase1 \
    --lane-script scripts/remote_archive_only_eval.sh \
    --label owv3_wave3_chain_eval \
    --anchor-dirs experiments/results/lane_g_v3_landed \
    --predicted-band 1.005 1.015 \
    --estimated-cost 1.50 \
    --max-dph 0.30 \
    --council-priority 1
```

**Step 2: Once SSH lands, manually SCP 6 archives** (~30s):
```bash
INSTANCE_PORT=<from launcher>
for arch in owv3_0043_bbr0p695_protect0p002_aggr1em05 owv3_0032_bbr0p7_protect0p002_aggr1em05 owv3_0076_bbr0p68_protect0p002_aggr1em05 owv3_0065_bbr0p685_protect0p002_aggr1em05 owv3_0120_bbr0p66_protect0p002_aggr1em05 owv3_0119_bbr0p66_protect0p0018_aggr1em05; do
    scp -P $PORT experiments/results/lane_g_v3_owv3_wave3_refinement_20260501/archives/${arch}.zip root@ssh3.vast.ai:/workspace/pact/${arch}.zip
done
```

**Step 3: Chain-eval each via remote_archive_only_eval.sh** — pattern from chain_eval.sh on 35955469 today.

## Custody chain (local)

`experiments/results/lane_g_v3_owv3_wave3_refinement_20260501/`:
- `archives/` (121 zips, 600KB-650KB each, ~75MB total)
- `wave3_chain_selection.json` (the 6 selected with predicted scores)

## Adversarial Grand Council pre-dispatch review

- **Shannon (LEAD):** "The protect=0.002 candidates push the rate-distortion frontier outward by trading SegNet's saturated headroom for PoseNet protection. Information-theoretic principled choice." **APPROVE.**
- **Dykstra (CO-LEAD):** "Convex-feasibility-wise: at bbr=0.65 protect=0.0013 we exited the linearization region (1.022 mild regress). At bbr=0.66 protect=0.002, the protect bump compensates — should stay inside Pareto-optimal frontier." **APPROVE WITH CONFIDENCE** — predict aggressive 1.005.
- **Yousfi:** "Distillation-spirit aside, the 6-candidate sweep is too small to triangulate the cliff precisely. RECOMMEND CONCURRENT broader Wave-4 at bbr ∈ {0.50, 0.55, 0.60} with protect={0.005, 0.010}." **APPROVE WITH ADDITION.**
- **Fridrich:** "Protect=0.002 is a 54% increase over R7's 0.0013 — that's a meaningful protection-strength dial that we haven't sampled before. Even at bbr=0.66, protect=0.002 should hold PoseNet better than R7's 0.0013 protection at bbr=0.7." **APPROVE.**
- **Contrarian:** "Eval noise is ±0.005 on contest-CUDA. Predicted band 1.005-1.012 is at-or-near the noise floor. Need a paired R7 re-eval on the same instance for rigor." **APPROVE WITH GUARDRAIL** — chain in R7 archive as 7th candidate to recompute baseline noise.
- **Hotz:** "Two consecutive sub-frontier wins. Don't break the streak by overthinking. SHIP THE 6." **APPROVE.**
- **Quantizr:** "0.68 distance to leader; need 2-3 more orthogonal stacks. Wave-3 is one orthogonal axis (further-tuned Fisher selection); the others (PD-V2, LCT, multi-pass) need bringing up in parallel." **APPROVE WITH RECOMMENDATION** — start PD-V2 + LCT bolt-on integration concurrently.

**VERDICT: 7/0 APPROVE Wave-3 dispatch + add R7 re-eval as 7th candidate; concurrent Wave-4 broader sweep recommended next tick.**

## What would change my mind (reactivation criteria)

- All 6 candidates land worse than R7 1.013 → R7 was already at the optimum, future gains require orthogonal axis (PD-V2, LCT, multi-pass)
- Best of 6 lands sub-1.000 [contest-CUDA] → first sub-1.0 of session, paradigm-shift moment
- Most aggressive (owv3_0120) lands sub-1.005 with PoseNet held → cliff is at bbr < 0.66 not 0.65, can push further
- Most aggressive shows PoseNet drift > 30% → cliff is RIGHT at bbr=0.66, R7 0.7 was already near-optimal

## Cross-refs

- `project_lane_g_v3_owv3_r7_LANDED_1_013_20260501.md` (the R7 baseline this builds on)
- `project_lane_g_v3_owv3_fisher_beta_LANDED_1_016_20260501.md` (β Fisher predecessor)
- `experiments/sweep_owv3_byte_plan.py` (the selector machinery)
- `scripts/remote_archive_only_eval.sh` (the chain-eval wrapper)
- `experiments/results/lane_g_v3_owv3_wave3_refinement_20260501/archives/` (the 121 archives, including the 6 selected)
