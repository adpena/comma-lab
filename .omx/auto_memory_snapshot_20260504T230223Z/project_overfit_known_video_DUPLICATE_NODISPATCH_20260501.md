# Lane overfit-known-video — DUPLICATE / NO-DISPATCH verdict

**Date**: 2026-05-01 ~13:30 UTC
**Subagent**: overfit-to-known-video lane
**Strategy memo**: `.omx/research/overfit_known_video_strategy_20260501.md`
**Lane registry id**: `lane_overfit_known_video` (Phase 3, L1 after memory_entry)
**Champion baseline**: owv3_0120 = 1.0024 [contest-CUDA RTX 4090], 617,410 B
**GPU spent**: $0
**Verdict**: DUPLICATE / NO-DISPATCH (council 7/0)

## Mandate

User parent-agent dispatched a "known-exact-video exploit" subagent with explicit guidance: "Other agents are working on NeRV codec, Joint-ADMM, orthogonal stack composition — DO NOT duplicate. You own the **overfit-to-known-video axis**." Suggested 5 approaches (frame-idx NN, per-frame residual, Rust decoder, RL allocation, PoseNet-aware truncation).

## Adversarial Grand Council review (7 voices)

1. **Shannon (LEAD)**: param-per-dim ratio of frame-idx-NN approach is 2e-4 → distortion floor too high vs renderer-with-temporal-context. Soft sensitivity weighting (β Fisher OWv3) is the rate-distortion-optimal form, ALREADY landed at 1.013.
2. **Dykstra (CO-LEAD)**: per-frame residual codec adds rate (50-200KB) without compensating distortion drop unless aggressively scorer-weighted; the Pareto frontier doesn't move.
3. **Yousfi**: PoseNet-aware truncation is literal UNIWARD inverse-steganalysis. Lane UNIWARD v8 already empirically tested it (1.14 advisory), SegNet detects sharp class-boundary truncation artifacts.
4. **Fridrich**: <30% of pixels per frame are score-relevant — but THIS IS WHAT THE RENDERER ALREADY LEARNS. The 211KB renderer.bin already encodes the score-relevant manifold.
5. **Carmack (grand)**: Rust decoder size ENTERS the rate term. A 50KB Rust binary + 200KB payload = 250KB total — same envelope as a python decoder of the same payload.
6. **Hassabis (grand)**: RL allocation sweep IS Lane G v3 OWv3 r6/r7/wave3 already in flight (champion came from this).
7. **Hotz**: ship the simplest version that beats baseline. The Phase 1 multi-day OWv3 sweep is doing this with bbr/protect/aggr knobs. Don't open a new dispatch axis when the existing one is delivering.
8. **Contrarian**: every prompt-suggested approach maps to an existing lane id. There is NO orthogonal-and-novel approach in the prompt that isn't already represented:
   - Approach A (frame-idx NN) → `lane_j_nwc` (Phase 1.5, L2 landed)
   - Approach B (residual) → covered by Phase 3 multi-pass + bit-level archive opt (`lane_multi_pass_inflate`, `lane_bit_level_archive_opt`)
   - Approach C (Rust decoder) → orthogonal but rate-neutral (decoder bytes count); council unanimous WEAK
   - Approach D (RL allocation) → `lane_rl_pufferlib_bandit` (Phase 3, L0) AND `lane_owv3_0120` (Phase 1, L2 — champion came from this exact pattern)
   - Approach E (sensitivity-weighted) → `lane_12_nerv_mask_codec` (Phase 2, L2) + β Fisher OWv3 (already landed 1.016) + `lane_sensitivity_map` (Phase 3, L0) + Lane UNIWARD v8 (1.14 advisory)
9. **Quantizr**: he leads at 0.33 with 88K-param FiLM CNN + odd-only mask trick. NO PUBLISHED sensitivity-weighted scheme. Frame2-only-with-warp IS NOVEL but maps to NeRV mask codec lane (Lane 12).

**Vote**: 7 NO-DISPATCH / 0 DISPATCH / 2 (Quantizr, Fridrich) suggest registering the frame2-only-mask-warp variant as a sub-lane of NeRV mask codec rather than a separate dispatch.

## Internal consistency check

What this verifier checked:
- ✓ `upstream/README.md` confirms single-video scoring (`videos/0.mkv`)
- ✓ `upstream/evaluate.sh` defaults to `public_test_video_names.txt` containing only `0.mkv`
- ✓ `upstream/videos/0.mkv` exists at 35.8MB on disk; `uncompressed_size` = sum-of-files = 37,545,489 B
- ✓ Champion archive contents: `unzip -l` shows renderer.bin (211KB) + masks.mkv (421KB) + optimized_poses.pt (15KB) = 649KB pre-zip / 617KB post-zip
- ✓ Score arithmetic verified: 100×0.00402 + sqrt(10×0.00356) + 25×0.01644 = 0.4020 + 0.1887 + 0.4110 = 1.0017 (matches reported 1.0024 within rounding)
- ✓ Archive byte entropy at 100% — no ZIP-level deflate headroom remains
- ✓ Lane registry audit shows Lane J-NWC, Lane 12 NeRV mask codec, Lane sensitivity_map, Lane RL pufferlib bandit, Lane OWv3 0120 ALL present → 5/5 prompt approaches duplicated

## What would change my mind

This NO-DISPATCH verdict is reactivated if:

1. **A duplicate lane empirically falsifies its approach** — e.g., Lane 12 NeRV mask codec returns "frame2-only mask trick raises seg by 3×, untenable" → then a NOVEL re-attack is justified.
2. **A genuinely orthogonal axis is discovered** — e.g., a YUV chroma channel that the SegNet ignores entirely (memory `feedback_uniward_texture_pattern` + Lane UNIWARD v8 cover this; needs CUDA confirm of v8's 1.14 first).
3. **The champion baseline regresses** — if owv3_0120 is invalidated by a re-eval drift > 0.02, the strategy stack must be re-derived.
4. **A new contest scoring path is announced** that breaks the single-video assumption — currently `upstream/README.md` reads "0.mkv is a 1 minute 37.5 MB dashcam video" with no hidden test set.
5. **Compute budget swap** — if the parent dispatcher releases $20+ to this subagent specifically for redundant validation, the OWv3 re-sweep at higher protect-grain values would be the highest-EV duplicate burn.

## Cost analysis

- **Saved**: $2-5 of GPU cost by NOT dispatching a duplicate of in-flight Phase 1/1.5/2/3 lanes.
- **Time spent**: ~25 min (memo + lane registration + this council review).
- **Net value**: orthogonal-axis registration prevents future subagents from re-running this analysis; lane gap explicitly documented at L1.

## Cross-references

- `.omx/research/overfit_known_video_strategy_20260501.md` (full strategy memo with per-approach council voting)
- `project_lane_g_v3_owv3_r7_LANDED_1_013_20260501.md` (current champion lineage)
- `project_lane_g_v3_owv3_fisher_beta_LANDED_1_016_20260501.md` (β Fisher = Approach E soft-form)
- `project_lane_j_nwc_landed_20260429.md` (Lane J-NWC = Approach A neural codec)
- `project_phases_2_3_4_design_implementation_math_provenance_20260429.md` (Lane 12 NeRV + Lane sensitivity_map roadmap = Approaches D and E)
- `project_lane_uniward_v8_harvested_1_14_advisory_20260429.md` (Approach E hard-form, advisory 1.14)
- `project_grand_council_final_designs_20260429.md` (frame2-only mask trick coverage)
