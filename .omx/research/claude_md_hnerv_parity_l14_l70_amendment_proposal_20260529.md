# CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L14-L70 Amendment Proposal (Slot DD)

`provenance: slot_dd_cross_pr_family_deep_research_l14_l70_20260529`
`status: operator-decision-pending per "iterate not force" standing directive`
`generated_at: 2026-05-29T07:55:00Z`
`source_memo: cross_pr_family_canonical_techniques_mining_L14_L70_20260529T075244Z.md`
`canonical_apparatus_mutation_chain_status: canonical equations + anti-pattern + posterior anchor LANDED via Phase E`
`mission_predicted_contribution: frontier_breaking`

## Why this proposal is operator-decision-pending (not applied directly)

Per CLAUDE.md "Design decisions — non-negotiable" + standing directive *"iterate on ultimate until grand council symposium approval then deploy don't force"*: CLAUDE.md amendments lifting NEW canonical L# numbered lessons are council-grade decisions requiring operator + inner-council sextet sign-off. This memo is the canonical AMENDMENT PROPOSAL surface — the operator routes to either (a) ratify-and-apply via canonical edit-CLAUDE.md subagent, (b) refine-then-apply, or (c) defer pending sister evidence.

Per CLAUDE.md "Council conduct" amendment 2026-05-19 4-co-lead structure (Shannon LEAD + Dykstra CO-LEAD + Rudin CO-LEAD + Daubechies CO-LEAD): the L14-L70 amendment touches CLAUDE.md non-negotiable; per Catalog #300 T3 grand-council deliberation tier with topical attendees (PR95-author canonical inner-council seat + Quantizr canonical adversarial seat + Selfcomp canonical PR #56 lead + Mallat-Daubechies-Rao-Ballard-Atick-Redlich-Tishby-Zaslavsky-Wyner topical for L43-L70 paradigms).

## Proposed amendment

Apply to CLAUDE.md "HNeRV / leaderboard-implementation parity discipline — NON-NEGOTIABLE, HIGHEST EMPHASIS" section, immediately after the existing 13 inviolable lessons.

### Insert at line ~360 of CLAUDE.md (after `13. **KILL/FALSIFIED is LAST RESORT.**` line)

```markdown
### Lessons L14–L42 — PR95-family canonical techniques (APPENDED 2026-05-29 per Slot DD cross-PR-family deep-research)

Each lesson L14-L42 mined from `experiments/results/public_pr95_intake_20260504_codex/profile_pr95_hnerv_muon_intake.md` + `experiments/results/public_pr100_intake_20260504_codex/source/submissions/hnerv_lc_v2/{hnerv_model.py,inflate.py,schema.py,sidecar.py}` + `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/{src/codec.py,src/model.py,inflate.py,inflate.sh,README.md}` + `experiments/results/public_pr103_intake_20260504_codex/source/submissions/hnerv_lc_ac/inflate.py`. Each lesson registered as canonical equation + sister canonical anti-pattern per Catalog #344.

14. **PR95 8-stage 29,650-epoch training curriculum.** stage1=CE (3k ep) → stage2=tau_softplus (5.65k) → stage3=smooth (1.5k) → stage4=QAT (500) → stage5=C1a-L7 (9k) → stage6=lambda_sweep (2k) → stage7=sigma_sweep (3k) → stage8=muon_finetune (5k). Each stage carries (loss_form, learning_rate, qat_active, c1a_lambda, sigma) tuple.

15. **Muon optimizer in final stage only.** 177,156 of 228,958 decoder params under Muon (77%); remaining 51,802 under AdamW; stages 1-7 all AdamW. Per-param-group optimizer assignment is canonical.

16. **C1a coder-aware regularization weight schedule (lambda 0.01 → 0.02).** Structural prior that biases decoder weights toward brotli-friendly distributions; starts stage 5 lambda=0.01, sweeps to 0.02 in stages 6-8.

17. **Sigma noise injection schedule (0.2 → 0.1).** Structural regularizer that simulates uint8 quantization roundtrip during training; sister of eval_roundtrip=True CLAUDE.md non-negotiable but distinct discipline.

18. **PixelShuffle + bilinear-skip + sin activation decoder.** 6 upsample stages from 6x8 to 384x512 native eval resolution; channel taper `[C, C, C, 0.75C, 0.58C, 0.5C, 0.5C]`. NeRF-style sin activation avoids dead-ReLU regions for single-video memorization.

19. **Per-frame-pair latent 28-d predicting 2 frames per latent.** 600 latents × 2 = 1200 contest frames; ~94% of archive bytes = decoder weights, ~6% = per-pair latents.

20. **Monolithic single-file 0.bin archive grammar.** 4-section (PR100) OR 8-section (PR101) length-prefixed grammar inside ZIP STORED `0.bin` member. NO separate ZIP members per Catalog #146 contest contract.

21. **Per-tensor byte-maps for entropy-friendly coding** (`zig` / `negzig` / `twos` / `off`). PR101 `DECODER_BYTE_MAPS = {tensor_idx → map_name}` for 4 specific tensors selected based on weight distribution post-quantization.

22. **CONV4_STORAGE_PERMS per-tensor permutation.** PR101 `CONV4_STORAGE_PERMS = {idx: perm}` for 13 specific Conv2d tensors; reorders axes before brotli compression; inverse perm applied at decode.

23. **Split brotli streams with explicit DECODER_STREAM_ENDS partition.** PR101 7 separate brotli streams (not one big stream); `DECODER_STREAM_ENDS = (1, 2, 22, 23, 26, 27, 28)`; each stream decoded independently.

24. **Raw LZMA latent coding** (FORMAT_RAW + FILTER_LZMA1 dict_size=4096). PR101 uses `LATENT_LZMA_FILTERS = [{id: FILTER_LZMA1, dict_size: 4096, lc: 3, lp: 0, pb: 0}]` instead of standard .xz format; saves bytes by stripping format headers.

25. **Temporal-delta uint8 latent coding with prefix-sum decode.** PR100/PR101 store latents as temporal deltas; prefix-sum reconstructs sequence; exploits temporal smoothness of dashcam video.

26. **Canonical Huffman length-vector ranked sidecar** (Wang-Rudin discipline). PR101 sidecar uses canonical Huffman codes with length-vector RANK encoding: instead of storing Huffman tree, store rank of length-vector among all Kraft-valid vectors. 607-byte `SIDECAR_HUFF_ENUM_LEN`.

27. **Combinatorial colex rank encoding for no-op positions.** PR101 encodes sidecar no-op positions via combinatorial colex rank: 3-byte `SIDECAR_NOOP_INFER_RANK_LEN` instead of N_PAIRS/8 bytes for bitmap.

28. **PR #98 decode-side channel-balance correction.** PR101 inflate.py post-processes decoded frames: subtract 1.0 from frame_0 RED channel, frame_0 BLUE channel, frame_1 GREEN channel; learned during training to compensate for a known scorer bias. **0 archive bytes, ~-0.0001 to -0.0005 score points.**

29. **fp16 scales per tensor for INT8 dequant.** PR100/PR101 store one fp16 scale per tensor (28 tensors = 56 bytes); `dequant = int8_code.astype(fp32) * fp16_scale`.

30. **Range/arithmetic coding via constriction.Categorical for specific tensors** (PR103 silver). PR103 silver-medal substitutes brotli with `constriction.stream.queue.RangeDecoder` + per-tensor `Categorical` histogram for 8 specific large tensors (`AC_INDICES = [0, 2, 4, 6, 8, 10, 12, 21]`); remaining 20 tensors stay brotli-encoded.

31. **Per-pair single-dim latent correction sidecar (255-sentinel no-op).** PR100/PR101 add a sidecar of `(u8 dim_idx, i8 delta_quantized)` per pair: `dim_idx=255` means no correction; `delta` scaled by 0.01; ~1.2KB sidecar encodes targeted fine-tune corrections per pair selected to minimize SegNet+PoseNet distortion. **This single technique contributes -0.001 to -0.003 score improvement (substrate-ceiling → medal-class jump).**

32. **brotli quality=11 max compression for sidecar.** PR100/PR101 use `brotli.compress(payload, quality=11)` (max) for all sidecar encoding.

33. **KL distillation with T=2.0 temperature for SegNet supervision.** Quantizr canonical (Quantizr 0.33 anchor); PR100 hnerv_lc_v2 also uses; sister of Hinton/Vinyals/Dean 2014 knowledge distillation paradigm.

34. **EMA decay 0.997 for weight EMA + 0.99 for codebook EMA.** Per CLAUDE.md "EMA — NON-NEGOTIABLE" non-negotiable; canonical Quantizr 0.33 anchor + sister A1 + PR95 + PR100 + PR101 all use. Apply only at eval time with snapshot+restore; archive bytes come from `ema.state_dict()` NOT `model.state_dict()`.

35. **Cosine LR schedule with warmup (per-stage rates).** PR95 stages 1-8 AdamW lr [1e-3, 1e-3, 1e-4, 1e-4, 3e-5, 3e-5, 3e-5, 1e-5] — per-stage LR schedule encoded in 8-stage curriculum.

36. **Deterministic reproducibility.** `torch.manual_seed(profile.seed)` + `numpy.random.seed` + `random.seed` + `torch.use_deterministic_algorithms(True)` where possible. Per CLAUDE.md "Canonical pipeline standard" non-negotiable.

37. **Hardware-aware numeric (TF32 / autocast fp16 / torch.compile).** Per Catalog #178 (TF32) + Catalog #172 (autocast fp16) + Catalog #179 (torch.compile). 4-6× wall-clock speedup; no direct score impact.

38. **no_grad eval-time memory hygiene.** Per Catalog #180 strict-from-byte-one. `torch.no_grad()` / `torch.inference_mode()` context wraps eval-time scorer forwards.

39. **Per-axis decomposition emission.** Per Catalog #356 (per-axis canonical Provenance per Catalog #323). Enables Pareto polytope intersection via Catalog #372 Dykstra Pareto polytope solver.

40. **Brotli quality + level + dict configuration tuning.** PR101 specific brotli config per-tensor + per-stream (not uniform default).

41. **Archive ZIP STORED (no compression) vs DEFLATED choice.** STORED is canonical for the outer `archive.zip` because the inner `0.bin` member is already brotli/LZMA pre-compressed; DEFLATED INFLATES total bytes by ~50-200. ZIP framing overhead ≈ 108 bytes.

42. **Per-pair mask grammar (per-frame vs per-pair vs per-region menu).** PR101 frame-exploit selector + PR110-OPT family (FEC6/FEC10); canonical bolt-on substrate for cross-archive composition; -7.66e-6 to -1.5e-4 per FEC stack (empirical anchors).

### Lessons L43–L70 — CROSS-PR-FAMILY canonical techniques (APPENDED 2026-05-29 per Slot DD Contrarian binding revision)

Each lesson L43-L70 mined from sister PRs + OUTSIDE-PR-95-family paradigms per Slot CC verdict Contrarian binding revision *"TRUE class-shift may be OUTSIDE PR-95-family"*. 10 of 28 are paradigm-level CLASS-SHIFT candidates beyond PR-95-family HNeRV lineage (L43, L44, L46, L47, L48, L49, L50, L52, L58, L67).

43. **Selfcomp (PR #56) grayscale-LUT analog mask paradigm.** 1.017-bpw block-FP weight self-compression + 94K-param SegMap. Selfcomp 0.38 SOLO anchor; ORTHOGONAL to PR-95-family HNeRV lineage. CLASS-SHIFT candidate.

44. **Block-FP quantization granularity (1.017 bpw per Selfcomp).** Enables sub-1-bpw weight storage; not HNeRV-compatible.

45. **Cool-Chic / C3 generative neural compression (Ballé hyperprior lineage).** Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L2 deferred-pending-export.

46. **Wyner-Ziv decoder-side side-information consumption (1976 source coding with side info theorem).** Per CLAUDE.md grand-council Wyner + DP1 Phase 2 + Catalog #319 deliverability gate. CLASS-SHIFT (sister Slot R synthesize_frame Atick-Redlich enabler).

47. **Atick-Redlich 1990 cooperative-receiver loss (predictive coding via I(X;T)/I(T;Y) bottleneck).** Per CLAUDE.md grand-council Atick + Redlich + Tishby memorial + Zaslavsky. Canonical paradigm for Z4/Z6/Slot R synthesize_frame substrates. CLASS-SHIFT.

48. **Rao-Ballard 1999 hierarchical predictive coding.** Per CLAUDE.md grand-council Rao + Ballard + Catalog #311 sister. Z5/Z6/Z7/Z8 canonical paradigm. CLASS-SHIFT.

49. **Hierarchical predictive coding canonical quadruple** (Rao-Ballard + Mallat wavelet + Hafner DreamerV3 + Wyner-Ziv). Per CLAUDE.md Catalog #312 + Z8 design memo Section 4.3. CLASS-SHIFT.

50. **Mallat wavelet + Daubechies compressive sensing multi-scale partition prior.** Per CLAUDE.md grand-council Mallat + Daubechies inner-council CO-LEAD + Catalog #277. Mallat 2009 + Daubechies-DeVore 2010 canonical. CLASS-SHIFT (Class A symposium predicted ΔS -0.02 to -0.04).

51. **Daubechies multi-scale hierarchical-coarse-gates-fine ordering.** Per CLAUDE.md Catalog #254 + Catalog #277.

52. **Tishby 2015 Information Bottleneck principle.** Per CLAUDE.md grand-council Tishby memorial + Zaslavsky living-voice. Tishby-Zaslavsky 2015 canonical. CLASS-SHIFT.

53. **Hinton knowledge distillation (T=2.0 temperature; the 2014 Hinton/Vinyals/Dean paper).** Per CLAUDE.md grand-council Hinton + Quantizr canonical direct user.

54. **van den Oord VQ-VAE codebook EMA (persistent N_c/m_c form; codebook decay 0.99).** Per CLAUDE.md grand-council van den Oord + EMA non-negotiable.

55. **Carmack-Hotz engineering-shortcuts paradigm.** NSCS06 Strip-Everything canonical (NSCS06 v7 44% improvement per CLAUDE.md "Substrate MUST be at OPTIMAL FORM" canonical example).

56. **Schmidhuber compression-as-intelligence + MDL prior.** Per CLAUDE.md grand-council Schmidhuber + MacKay memorial.

57. **MacKay arithmetic coding + Bayesian inference + Density-network framework.** Per CLAUDE.md grand-council MacKay memorial seat. MacKay 2003 ITILA reference.

58. **Ballé 2018 entropy bottleneck + scale hyperprior + GDN nonlinearity.** Per CLAUDE.md grand-council Ballé inner-seat. CLASS-SHIFT enabler if neural compression substrate lands.

59. **Fridrich UNIWARD adaptive embedding (errors in textured regions undetectable).** Per CLAUDE.md "Fridrich inverse steganalysis" + Catalog #259. PR110-OPT-7 deferred per Slot K KILL → DEFER per Catalog #313 30-day reactivation window.

60. **Fridrich STC (syndrome-trellis coding) parity-check codes.** Per CLAUDE.md grand-council Tomáš Filler + STC canonical (Filler-Fridrich 2011).

61. **Yousfi steganalysis surgery (EfficientNet stride-2 stem blind spots).** Per CLAUDE.md "Exact scorer architectures" + Yousfi DDELab repos.

62. **Boyd ADMM (Alternating Direction Method of Multipliers).** Per CLAUDE.md grand-council Boyd inner-council CO-LEAD. Boyd-Vandenberghe 2004 + Boyd ADMM 2011 canonical. Foundation for L63 + Catalog #372 Dykstra Pareto polytope solver.

63. **Dykstra alternating-projections feasibility.** Per CLAUDE.md grand-council Dykstra inner-council CO-LEAD + Catalog #296 + #372. Dykstra 1983 canonical.

64. **Rudin interpretable ML (falling-rule-lists + SLIM + GOSDT + Rashomon ensemble).** Per CLAUDE.md grand-council Rudin inner-council CO-LEAD + Catalog #273-#278. Wang-Rudin 2015 + Ustun-Rudin 2016 + Lin-Zhong-Hu-Hu-Rudin-Seltzer 2020 canonical.

65. **Karpathy let-compute-speak engineering practitioner discipline.** Per CLAUDE.md grand-council Karpathy.

66. **Time-Traveler protégé (canonical identity resolved to Rudin 2026-05-19).** Per CLAUDE.md grand-council Time-Traveler protégé seat.

67. **Selfcomp / szabolcs-cs grayscale-LUT analog mask + 88K-94K SegMap (PR #56 lead).** Sister of L43; HARD-EARNED (Selfcomp 0.38 SOLO anchor). CLASS-SHIFT.

68. **Hassabis strategic-research + DeepMind cross-domain breadth.** Per CLAUDE.md grand-council Hassabis.

69. **Demis-Hassabis-aligned 4-day-deadline tradeoff systemization.** Per CLAUDE.md grand-council Hassabis. Deadline-mode discipline.

70. **PR95-author canonical inner-council seat (added 2026-05-19).** Per CLAUDE.md "Design decisions" 2026-05-19 inner-council expansion. Canonical knowledge of May 4 2026 race-mode rigor inversion.
```

## Operator-routable refinement points

1. **L# numbering convention** — should L14+ be `L14-L42` (PR-95-family) + `L43-L70` (CROSS-PR-FAMILY) as proposed OR collapsed to single sequence L14-L70? Proposal: keep the 2-block split because the canonical-vs-CROSS distinction matters for Class A/D scope-locks per Phase D Slot CC verdict.

2. **CLASS-SHIFT subset highlighting** — should the 10 CLASS-SHIFT candidates (L43, L44, L46, L47, L48, L49, L50, L52, L58, L67) be promoted to a separate "INVIOLABLE CLASS-SHIFT lessons" subsection? Per operator binding directive *"continue MLX first and focused on score lowering and why we haven't produced on original frontier score yet"* the CLASS-SHIFT enumeration directly addresses the WHY-FRONTIER question.

3. **Per-lesson canonical equation backfill** — Phase E landed 3 canonical equations (meta + Class A + Class D); should EACH of L14-L70 land its own canonical equation per Catalog #344 sister discipline? Proposal: DEFER per "iterate not force" — only the 3 canonical equations land in same commit batch; per-lesson backfill is sister-subagent operator-routable.

4. **Empirical verification status per Catalog #363** — every L14-L70 lesson should carry `empirical_verification_status` ∈ `{VERIFIED_VIA_SOURCE_INSPECTION, VERIFIED_VIA_EMPIRICAL_ANCHOR, INFERRED_FROM_DOMAIN_LITERATURE, ASSUMED_AWAITING_VERIFICATION}`. Default per the canonical Slot DD evidence: L14-L42 = `VERIFIED_VIA_SOURCE_INSPECTION` (PR intake source); L43-L70 = `VERIFIED_VIA_SOURCE_INSPECTION` for CLAUDE.md grand-council references + `INFERRED_FROM_DOMAIN_LITERATURE` for paradigm-level techniques.

5. **8th forbidden pattern (the "research-substrate trap") amendment** — should the 8th forbidden pattern's enumeration be extended to reference L14-L70? Proposal: yes, extend at canonical-amendment time with reference to "L14-L42 PR-95-family parity discipline + L43-L70 CLASS-SHIFT exploration".

## Closure

This proposal is **operator-decision-pending**. The canonical apparatus mutation chain (3 canonical equations + 1 canonical anti-pattern + 1 canonical posterior anchor) LANDED in same commit batch via Phase E execution — the registry serves as queryable structural foundation for the amendment even before CLAUDE.md edit lands. mission_predicted_contribution=`frontier_breaking`.
