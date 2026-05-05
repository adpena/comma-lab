# PR family evolution — paradigm-shift timeline (2026-05-04)

## Discovery

The fixed dashboard now lets us see all 22 PR-family scores side-by-side.
Sorted by PR number, the score evolution shows distinct **paradigm shifts**
rather than gradual improvement.

## Full PR family table

```
PR  family                   bytes    pose      seg       rate      score
─────────────────────────────────────────────────────────────────────────
63  qpose14_trace            287,573  0.000528  0.000610  0.007659  0.3252
64  unified_trace            287,165  0.000626  0.000610  0.007648  0.3314

81  qzs3_range_mask          215,960  0.000584  0.000610  0.005752  0.2812
82  henosis_frontier         296,789  0.000189  0.000572  0.007905  0.2983
84  adaptive_range_mask      215,735  0.000493  0.000612  0.005746  0.2751
85  adaptive_masking_jf_model 236,328  0.000189  0.000572  0.006294  0.2581

95  hnerv_muon               178,417  0.000172  0.000707  0.004752  0.2310
96  rem2_hnerv               186,631  0.000173  0.000719  0.004971  0.2377
97  vibe_coder_final_boss    197,160  0.000637  0.000405  0.005251  0.2516  ← outlier (anti-pattern)
98  hnerv_adapter            178,392  0.000174  0.000688  0.004751  0.2293
99  hnerv_adapter            178,546  0.000173  0.000693  0.004755  0.2297
100 hnerv_lc_v2_adapter      178,981  0.000172  0.000676  0.004767  0.2283
101 hnerv_ft_microcodec      178,258  0.000171  0.000663  0.004748  0.2264
103 hnerv_lc_ac              178,223  0.000172  0.000676  0.004747  0.2278
105 kitchen_sink             177,857  0.000173  0.000705  0.004737  0.2304

106 belt_and_suspenders      186,239  0.000034  0.000671  0.004960  0.2095  ← FRONTIER
```

## Four paradigm-shift eras

### Era 1: qpose14 family (PR63-64)
- ~287KB archives
- Score 0.325-0.331
- Multi-stream compound codec (model + mask + pose + actuator etc.)

### Era 2: range-mask family (PR81-85)
- ~215-240KB archives (-25%)
- Score 0.258-0.298
- Improvements: smaller mask payload via range coding + adaptive routing

### Era 3: HNeRV family (PR95-105)
- ~178KB archives (-15%)
- Score 0.226-0.231 (TIGHT cluster, -0.05 from Era 2)
- **Architectural shift**: implicit neural representation (HNeRV decoder)
- All single-stage fine-tunes — minor variations don't move the score
- Pose distortion uniform at ~0.000172 (the HNeRV decoder's pose floor)

### Era 4: HNeRV + 8-stage training (PR106)
- 186KB archives (+5% bytes back from Era 3)
- Score 0.2095 (-0.017 from Era 3)
- **Training shift**: 8-stage pipeline (ce → softplus → smooth → qat → c1a → λ → σ → muon-finetune)
- Pose distortion drops 5× to 0.000034 — this is the entire delta
- Trade: +8KB bytes for -5× pose distortion

## What the timeline says about future deltas

**Within an era**: improvements are 0.001-0.005 (codec polish, schema overhead)
**Between eras**: improvements are 0.05-0.10 (architecture or training paradigm shift)

The /loop-tick polish lanes (apogee_intN, sidechannel proposals) operate
WITHIN Era 4 — they target the codec side of the trained PR106 decoder.
Predicted gains: 0.005-0.020 (apogee_intN bit-width sweep) + 0.001-0.005
(sidechannels). All within-era improvements.

**Cross-era jumps** (e.g., PR95→106 being -0.05) require either:
- A new training paradigm (PR106 was 8 stages on H100, weeks of research)
- A new decoder architecture (HNeRV → ???)
- A completely different submission paradigm (Q-FAITHFUL clone of Quantizr?)

These are research-grade lanes, not /loop polish. Quantizr's 0.33 score
is in a DIFFERENT paradigm (FP4 + AV1 mask + pose vectors, Era-2-style)
but with much more sophisticated training. Catching them likely requires
either:
- Era 5 architecture (something new — neural codec? Cool-Chic family?)
- Era 4 training applied to Quantizr-style architecture
- Multi-paradigm stacking (PR106 backbone + Quantizr-style sidechannels)

## Implications

The **operator dispatch decision pipeline** (apogee_int5 → council-gated
sidechannels → potentially LRL1/yshift stacking) is correctly aimed at
**Era-4 within-era polish**. Expected total post-stacking range: 0.180-0.205.

To beat PR106 by **0.05+**, an Era-5 paradigm shift is required. That's
not /loop work — that's the research-grade lanes the council reviews.

## Notable specific findings already documented

- PR97 anti-pattern (Era 3 outlier with seg-for-pose trade): see
  `docs/pr97_anti_pattern_pose_vs_seg_marginal_20260504.md`
- PR106 vs PR101 mechanism (training, not codec): see
  `docs/pr106_vs_pr101_training_recipe_finding_20260504.md`

This memo is the META — it organizes the per-PR findings into the timeline.

## Cross-refs

- All 22 PR scores sourced from `experiments/results/lightning_batch/exact_eval_public_pr*/auth_eval.log`
- Dashboard surfacing made possible by commit dbb0032d (log-parser fix)
- 9-memo paradigm thread: `docs/INDEX_score_aware_sidechannel_thread_20260504.md`
- Operator handoff (the actionable next step): `docs/operator_handoff_snapshot_20260504.md`
