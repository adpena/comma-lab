# Codex Findings: Tilde Research Parallax NeRV Intake

written_at_utc: 2026-06-03T17:42:29Z
lane_id: lane_tilde_research_parallax_nerv_intake_20260603
agent: codex
branch_observed: main
primary_scope: factual evidence intake and adversarial leverage review
write_scope: .omx/research/codex_findings_tilde_research_parallax_nerv_intake_20260603T174229Z_codex.md
evidence_manifest_path: null
large_downloads_or_clones: false
implementation_files_touched: false
score_claim: false
frontier_score_claim: false
rank_or_kill_eligible: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
research_only: true

## Preflight Status

- Read `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, the top of `~/.claude/projects/-Users-adpena-Projects-pact/memory/MEMORY.md`, the current `reports/latest.md`, recent Codex NeRV/SNeRV/HiNeRV memos, and `.omx/state/subagent_progress.jsonl`.
- Confirmed this lane is already registered at L0 in `.omx/state/lane_registry.json`.
- Observed unrelated dirty worktree state in `.omx/state/lane_maturity_audit.log` and `.omx/state/lane_registry.json`; left it untouched.
- Did not stage, revert, dispatch, train, clone, or download large artifacts.
- Used GitHub/web/Hugging Face/arXiv APIs and raw small files only.

## Current Pact Anchors Consumed

- PR95/HNeRV is the control arm, not a generic HNeRV label. It remains the baseline to beat and includes its own curriculum, QAT, Muon final stage, and tight export/codec discipline.
- HiNeRV blockers remain source-faithful official controls and byte evidence: hierarchical temporal/local feature grid + trilinear interpolation + ConvNeXt-like blocks, patch/frame equivalence, QuantNoise/bitstream/entropy coder path, trained section-byte measurement, score-valued decoder-weight waterfill, hard-pair/tail gating, and receiver-closed full-600 archive/runtime proof before score authority.
- SNeRV blockers remain official MFU/HFR/TUB parity and native export binding, not just local analogue quality. The local receiver-safe temporal/LF features are useful only when charged and receiver-real; explicit LF bytes remain the dominant rate bottleneck until a representation genuinely reduces or predicts them.
- All Parallax/Wall/Aurora claims below are false-authority research intake. They do not convert to CPU/CUDA/MLX score claims.

## Source Inventory

Provenance distinction: Tilde Research's official GitHub org is `tilde-research`, but the public Parallax implementation and Parallax NanoGPT harness found in this intake live under `Yifei-Zuo/*`, not under the Tilde org. The Parallax paper includes Tilde-linked authors/contributors, and the HF/arXiv/GitHub surfaces are still relevant primary evidence, but they should not be reported as `tilde-research` org release assets.

| id | source | url | primary? | owner/authors | date/sha observed | relevance verdict |
|---|---|---|---:|---|---|---|
| tilde_site | Tilde Research official site / research index | https://tilderesearch.com/research | yes | Tilde Research | page lists research releases through Wall Attention `6.2.26`; fetched 2026-06-03 | official identity and current publication surface |
| tilde_github_org | GitHub org `tilde-research` | https://github.com/tilde-research | yes | GitHub org name `Tilde`, blog `tilderesearch.com`, twitter `tilderesearch`, 10 public repos | API updated_at `2026-06-02T22:04:28Z` | official OSS identity |
| tilde_hf_org | Hugging Face org `tilde-research` | https://huggingface.co/tilde-research | yes | Tilde Research team | HF page links official site and GitHub; API showed models `tilde-research/aurora-1.1B`, `tilde-research/sieve_coding` | identity + model artifacts, not video-codec relevant |
| parallax_arxiv | `Parallax: Parameterized Local Linear Attention for Language Modeling` | https://arxiv.org/abs/2605.29157 | yes | Yifei Zuo, Dhruv Pai, Zhichen Zeng, Alec Dewulf, Shuming Hu, Zhaoran Wang | submitted 2026-05-27, v1 2026-05-27 22:50:44 UTC | core Parallax paper |
| parallax_hf_paper | HF paper page for arXiv 2605.29157 | https://huggingface.co/papers/2605.29157 | secondary aggregation with author comment | same as arXiv; submitted by Yifei Zuo | published May 27, submitted May 29; links GitHub | useful aggregation and code pointer |
| parallax_repo | Official Parallax implementation | https://github.com/Yifei-Zuo/Parallax | yes | Yifei-Zuo; README authors Yifei Zuo, Dhruv Pai, Zhichen Zeng, Alec Dewulf, Shuming Hu, Zhaoran Wang | latest commit `2d0ac304bf7492dc5bbf9dcb1be3a23eee83edc5` at 2026-06-03T01:07:51Z; MIT; 54 stars; no releases; open issue #1 asks for HF model release | OSS runnable for CUDA/Triton attention kernels, not a Pact archive path |
| parallax_nanogpt | Parallax modded-NanoGPT harness | https://github.com/Yifei-Zuo/modded-nanogpt-plx | yes | Yifei-Zuo | latest commit `7698686df679a7990cf91571df64042c30168d5c`; MIT; no releases; repo size ~79,908 KB | LLM training evidence and optimizer interactions, no byte-closed video decoder |
| flashlla_repo | Local Linear Attention kernel predecessor | https://github.com/Yifei-Zuo/FlashLLA | yes | Yifei-Zuo | latest commit `436af91eda4d9bacbb70635cbcbb1105c5efbae9`; Apache-2.0; no releases | Parallax dependency lineage; still LLM CUDA kernel work |
| aurora_blog | `Aurora: A Leverage-Aware Optimizer for Rectangular Matrices` | https://blog.tilderesearch.com/blog/aurora | yes | Alec Dewulf, Dhruv Pai, Li Yang, Ashley Zhang, Ben Keigwin | blog date 2026-05-05 | optimizer relevance to Muon/PR95 final-stage controls |
| aurora_repo | `tilde-research/aurora-release` | https://github.com/tilde-research/aurora-release | yes | Tilde Research | latest commit `7303d8cb9999d735cb12c921f3651f04bf362524`; MIT; no releases; closed PR #1 dtype mismatch fix | strongest immediate leverage candidate, but only as optimizer smoke |
| wall_blog | `Wall Attention: Length Generalization With Diagonal Gates` | https://blog.tilderesearch.com/blog/wall-attn | yes | Dhruv Pai, Timor Averbuch, Ashley Zhang, Ben Keigwin, Alec Dewulf | blog date 2026-06-02; releases reference Triton kernels | per-channel temporal decay idea; direct code is LLM attention kernels |
| wall_repo | `tilde-research/wall-attention-release` | https://github.com/tilde-research/wall-attention-release | yes | Tilde Research | latest commit `ed8f2a6549dacdbfe273c52796ef9d33468884c8`; no releases; no issues | gated decay reference; not directly Pact-runnable |
| wall_fla_repo | `tilde-research/wall-flash-linear-attention` fork | https://github.com/tilde-research/wall-flash-linear-attention | yes | Tilde Research / FLA-derived | latest commit `0c03d95ff74176f9d8ec7a31d95aaa98829bfe12`; no releases; no issues | dependency-heavy kernel fork; not a codec |
| nitrobrew_repo | `tilde-research/nitrobrew-release` | https://github.com/tilde-research/nitrobrew-release | yes | Tilde Research | pushed 2026-04-28; Apache-2.0 | LLM distillation/logit throughput, not relevant to Pact NeRV byte-closed work |

Date note: intake date is 2026-06-03. Some Tilde research index/API rows include future-looking public metadata relative to the intake date; those were recorded as source metadata only, not as proof of chronological priority.

## Parallax Technical Read

Parallax is not visual parallax, multiview rendering, scene-motion modeling, or a 3D/video representation. The source-faithful name is `Parallax: Parameterized Local Linear Attention for Language Modeling`.

The paper and implementation define an LLM attention primitive. It upgrades Local Linear Attention by adding an extra query-like projector/probe over KV covariance, removes the prior LLA numerical solver, and ships hardware-aware CUDA/Triton/CuTeDSL kernels. The raw PyTorch reference computes two score tensors, one from `q dot k` and one from `r dot k`, then combines weighted value sums with a normalization correction:

- `q`, `r`: query-like tensors.
- `k`, `v`: KV tensors.
- causal masking and optional sliding-window support.
- output shape matches attention output, not an image/video frame.

Implementation surface:

- `Yifei-Zuo/Parallax` package name `parallax`, Python `>=3.12,<3.13`.
- Core dependencies: `torch==2.9.1`, `triton==3.5.1`, `numpy`.
- Optional decode extra: `nvidia-cutlass-dsl==4.1.0` and `nvidia-cutlass-dsl-libs-base`; README says current kernels are developed/tested on NVIDIA Hopper GPUs.
- Bench extra pulls `flash_attn>=2.8.3`, `pytest`, `psutil`, `rich`.
- Public entry points: `parallax_func`, `parallax_fwd`, `parallax_bwd`, `parallax_reference`, optional `parallax_decode`.

Training/inference loop:

- Training path replaces scaled-dot-product attention in transformer blocks with Parallax attention.
- Decode path remains KV-cache based; Parallax README explicitly says Parallax/LLA are not linear-complexity attention and still require KV cache for decoding.
- The modded-NanoGPT harness applies Parallax as a patch on language-model speedrun scripts by adding `self.r = Linear(dim, hdim)`, deriving `r = self.r(x)`, and replacing `F.scaled_dot_product_attention(...)` with `parallax_func(q, r, k, v, scale)`.

Compression-relevant payloads:

- No image/video frames, DWT bands, NeRV coordinate fields, frame-pair latents, decoder bitstream grammar, or contest `inflate.sh` runtime.
- Payloads would be ordinary model weights plus CUDA/Triton kernel code and any learned transformer state. In Pact scoring terms, any such weights/runtime would need a new charged archive/runtime grammar.
- There is no byte-closed archive path, receiver proof, modelsize measurement, scorer replay, or CUDA/CPU contest axis evidence.

## Direct Comparison To Pact Priorities

### PR95/HNeRV Control Arm

Parallax does not replace PR95/HNeRV. It is attention-architecture work on token sequences. The useful overlap is narrower: Parallax and the NanoGPT harness reinforce that Muon/architecture co-design can matter, and the harness reports Parallax improvements when paired with optimizers such as SOAP-H, DynMuon, Aurora, and Muon-like baselines. That is optimizer/control evidence, not a rate-distortion claim for Pact.

Best Pact use: treat `aurora-release` and the Parallax harness as optimizer-smoke inputs for the existing PR95/Muon final-stage control machinery. Do not compare Parallax LLM step reductions to Pact score.

### HiNeRV

HiNeRV needs source parity first: official feature-grid/ConvNeXt/trilinear controls, patch-index parity, QuantNoise/bitstream/entropy path, and trained byte-section measurement. Directly importing Parallax attention would create a non-source-faithful analogue and likely add runtime/dependency burden before the current parity blockers close.

Possible leverage is conceptual and narrow:

- gated local retention over temporal/local feature grids;
- learned per-channel temporal gates for hard-pair tail focus;
- optimizer interaction with Muon/Aurora on rectangular decoder matrices.

All three require a Pact-specific, charged, receiver-visible implementation before they can matter.

### SNeRV

SNeRV's urgent bottleneck is not generic sequence modeling; it is LF payload rate and official MFU/HFR/TUB parity. Parallax itself is not source-faithful to SNeRV. Wall Attention is a closer conceptual hint because it implements per-channel, per-timestep multiplicative decay with a PyTorch reference and Triton kernels, but it is still an LLM attention primitive.

Best Pact use: after official MFU/HFR/TUB parity or in a clearly labelled side smoke, test a tiny receiver-visible learned temporal gate for LF/TUB prediction. It must be charged as metadata/weights and compared against explicit LF bytes. A local MLX/no-op advisory feature is not sufficient; the prior SNeRV no-op class is exactly the bug to avoid.

## Opportunity Rows

```json
[
  {
    "id": "tilde_aurora_pr95_hinerv_optimizer_smoke",
    "classification": "timing_smoke_candidate",
    "target_stack": "HiNeRV/HNeRV",
    "source_ids": ["aurora_blog", "aurora_repo", "parallax_nanogpt"],
    "expected_mechanism": "Aurora is a Muon-adjacent optimizer for rectangular matrices that tries to keep row leverage/update mass uniform while preserving polar precision; Pact decoder/MLP matrices are exactly where final-stage Muon sensitivity may matter.",
    "cheapest_faithful_smoke": "Add a plan-only optimizer kind in the existing NeRV optimizer-control surface, then run a 1-2 epoch local/advisory timing-convergence smoke against an already-small PR95/HiNeRV training slice before any long campaign queue admission.",
    "concrete_code_surface": [
      "src/tac/analysis/nerv_long_training_campaign_plan.py",
      "src/tac/substrates/_shared/mlx_score_aware/",
      "src/tac/analysis/nerv_candidate_feedback.py"
    ],
    "required_evidence_axis": "[macOS-MLX research-signal] or local timing telemetry only, followed by receiver-closed archive/runtime and exact contest-CPU/CUDA before authority",
    "blockers": [
      "not_source_faithful_to_pr95_until_controlled_against_existing_muon_final_stage",
      "aurora_not_integrated_with_mlx_score_aware_optimizer_contract",
      "no_trained_byte_section_measurement",
      "no_receiver_closed_archive_runtime"
    ],
    "why_it_could_move_score_rate_segnet_posenet": "If final-stage decoder convergence improves at the same modelsize/byte ceiling, it could lower SegNet/PoseNet distortion without adding archive bytes except optimizer metadata outside runtime.",
    "false_authority": {
      "score_claim": false,
      "promotion_eligible": false,
      "ready_for_exact_eval_dispatch": false
    }
  },
  {
    "id": "tilde_wall_attention_snerv_lf_temporal_gate_smoke",
    "classification": "timing_smoke_candidate",
    "target_stack": "SNeRV",
    "source_ids": ["wall_blog", "wall_repo"],
    "expected_mechanism": "Wall Attention's per-channel multiplicative decay is a source for a tiny learned temporal retention/gate over LF/TUB inputs; a successful Pact variant would reduce explicit LF payload by predicting more LF structure from neighboring frames.",
    "cheapest_faithful_smoke": "Build a receiver-visible opt-in LF/TUB temporal-gate feature with all learned gate parameters byte-charged, then run a bounded pair/hard-pair smoke that reports LF bytes, reconstructed LF error, SegNet/PoseNet component deltas, and receiver replay status.",
    "concrete_code_surface": [
      "src/tac/substrates/snerv_inverse_steg_carrier/official_tub.py",
      "src/tac/substrates/snerv_inverse_steg_carrier/",
      "src/tac/analysis/snerv_lf_payload_archive_recode.py",
      "src/tac/analysis/nerv_long_training_campaign_plan.py"
    ],
    "required_evidence_axis": "receiver-real SNeRV archive/runtime proof for any byte claim; local MLX advisory allowed only for timing/fit selection",
    "blockers": [
      "SNERV_OFFICIAL_MFU_HFR_TUB_PARITY_PROOF_missing",
      "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload",
      "learned_gate_bytes_not_yet_charged",
      "full600_or_hardpair_distortion_replay_missing"
    ],
    "why_it_could_move_score_rate_segnet_posenet": "SNeRV LF bytes currently dominate rate. A byte-charged temporal LF predictor can only help if it removes more LF payload bytes than its parameters/metadata add while preserving scorer-sensitive low-frequency pose/seg structure.",
    "false_authority": {
      "score_claim": false,
      "promotion_eligible": false,
      "ready_for_exact_eval_dispatch": false
    }
  },
  {
    "id": "tilde_parallax_hinerv_feature_grid_gate",
    "classification": "research_only_blocked",
    "target_stack": "HiNeRV",
    "source_ids": ["parallax_arxiv", "parallax_repo", "parallax_nanogpt"],
    "expected_mechanism": "A Parallax-like query/probe gate could condition temporal feature-grid retention or hard-pair tail specialization.",
    "cheapest_faithful_smoke": "None before official HiNeRV feature-grid/ConvNeXt/patch/QuantNoise controls are source-bound. After parity, a tiny opt-in gate could be tested as a new non-source baseline with charged parameters.",
    "concrete_code_surface": [
      "src/tac/analysis/nerv_source_parity_contract.py",
      "src/tac/analysis/nerv_long_training_campaign_plan.py"
    ],
    "required_evidence_axis": "source-parity contract first; advisory training telemetry second; receiver-closed exact axis last",
    "blockers": [
      "hi_nerv_official_feature_grid_convnext_trilinear_missing_or_unproven",
      "hi_nerv_prune_quantnoise_receiver_bitstream_pipeline_missing_or_unproven",
      "would_create_non_source_faithful_analogue_before_current_parity_blockers_close"
    ],
    "why_it_could_move_score_rate_segnet_posenet": "The mechanism could improve hard-pair reconstruction with small parameter overhead, but there is no evidence yet that it beats source-faithful HiNeRV controls under Pact byte accounting.",
    "false_authority": {
      "score_claim": false,
      "promotion_eligible": false,
      "ready_for_exact_eval_dispatch": false
    }
  },
  {
    "id": "tilde_parallax_direct_kernel_port",
    "classification": "not_relevant",
    "target_stack": "cross-stack",
    "source_ids": ["parallax_repo", "flashlla_repo"],
    "expected_mechanism": "Directly use Parallax/LLA kernels in Pact runtime.",
    "cheapest_faithful_smoke": "No faithful smoke recommended.",
    "concrete_code_surface": [],
    "required_evidence_axis": "not applicable",
    "blockers": [
      "torch_2_9_1_triton_3_5_1_cuda_dependency",
      "optional_decode_requires_hopper_sm90_cutlass_dsl",
      "token_attention_output_not_video_decoder",
      "no_byte_closed_archive_or_inflate_contract",
      "runtime_dependency_burden_not_contest_faithful"
    ],
    "why_it_could_move_score_rate_segnet_posenet": "No credible direct path.",
    "false_authority": {
      "score_claim": false,
      "promotion_eligible": false,
      "ready_for_exact_eval_dispatch": false
    }
  },
  {
    "id": "tilde_parallax_nanogpt_training_records",
    "classification": "source_parity_input",
    "target_stack": "HiNeRV/HNeRV",
    "source_ids": ["parallax_nanogpt"],
    "expected_mechanism": "The harness reports Parallax step reductions under optimizer-controlled LLM speedrun scripts, especially with SOAP-H/DynMuon/Aurora. This is evidence that attention/optimizer co-design can matter under tight training budgets.",
    "cheapest_faithful_smoke": "Use only as optimizer-control prior for a Pact Aurora/Muon smoke; do not import the NanoGPT training stack.",
    "concrete_code_surface": [
      "src/tac/analysis/nerv_long_training_campaign_plan.py",
      "src/tac/analysis/nerv_candidate_feedback.py"
    ],
    "required_evidence_axis": "Pact-local training telemetry; no contest axis until archive/runtime proof",
    "blockers": [
      "FineWeb_cross_entropy_not_Pact_score",
      "8xH100_LLM_speedrun_not_contest_runtime",
      "no_archive_byte_accounting"
    ],
    "why_it_could_move_score_rate_segnet_posenet": "Only by improving optimizer choice for existing NeRV decoder training, not by transferring LLM results directly.",
    "false_authority": {
      "score_claim": false,
      "promotion_eligible": false,
      "ready_for_exact_eval_dispatch": false
    }
  },
  {
    "id": "tilde_wall_attention_direct_import",
    "classification": "not_relevant",
    "target_stack": "cross-stack",
    "source_ids": ["wall_repo", "wall_fla_repo"],
    "expected_mechanism": "Directly import Wall Attention kernels.",
    "cheapest_faithful_smoke": "No direct import smoke recommended.",
    "concrete_code_surface": [],
    "required_evidence_axis": "not applicable",
    "blockers": [
      "torch_triton_flash_linear_attention_dependency_stack",
      "CUDA_kernel_attention_runtime_not_Pact_decoder_grammar",
      "no_byte_closed_archive_path",
      "no_video_or_frame_pair_representation"
    ],
    "why_it_could_move_score_rate_segnet_posenet": "No direct scorer path; only the per-channel decay idea is worth translating into a tiny charged Pact-native gate.",
    "false_authority": {
      "score_claim": false,
      "promotion_eligible": false,
      "ready_for_exact_eval_dispatch": false
    }
  },
  {
    "id": "tilde_nitrobrew_and_hf_models",
    "classification": "not_relevant",
    "target_stack": "cross-stack",
    "source_ids": ["nitrobrew_repo", "tilde_hf_org"],
    "expected_mechanism": "LLM distillation/model artifacts.",
    "cheapest_faithful_smoke": "No Pact smoke recommended.",
    "concrete_code_surface": [],
    "required_evidence_axis": "not applicable",
    "blockers": [
      "LLM_distillation_not_video_compression",
      "model_artifacts_not_byte_closed_Pact_decoder",
      "no_HiNeRV_HNeRV_SNeRV_source_parity_signal"
    ],
    "why_it_could_move_score_rate_segnet_posenet": "No credible near-term mechanism.",
    "false_authority": {
      "score_claim": false,
      "promotion_eligible": false,
      "ready_for_exact_eval_dispatch": false
    }
  }
]
```

## Negative Findings

- `Parallax` is a name collision for this lane. The Tilde/Yifei Parallax work is not geometric parallax, multiview synthesis, RAFT/ego-motion, camera pose, neural radiance fields, or a 3D scene representation.
- The Parallax repo has no GitHub release assets and no released Parallax model weights as of this intake; issue #1 asks for Hugging Face model release.
- Parallax's current runtime story is CUDA/Triton and optional Hopper SM90 CuTeDSL decode. That is far from a small deterministic contest `inflate.sh` runtime.
- The Parallax and Wall repositories are kernel packages for LLM attention, not byte-closed codec systems. No ZIP member grammar, no archive byte accounting, no modelsize ladder, no receiver replay, and no exact scorer path exist.
- Wall Attention is slick and technically adjacent to temporal decay, but direct import would create the same dependency/runtime problem as Parallax. Only the tiny per-channel decay idea is worth translating into Pact-native byte-charged code.
- Aurora is useful only as a controlled optimizer candidate. It is not PR95 source parity, not a score claim, and not a reason to disturb current source-faithful HiNeRV/SNeRV blockers.
- NitroBrew, Sieve, Activault, MoMoE, NSA, and HF LLM/model artifacts did not reveal a credible byte-closed HiNeRV/HNeRV/SNeRV action in this pass.

## Machine-Readable Blocker Rows

```json
[
  {
    "blocker_id": "tilde_parallax_not_visual_parallax",
    "severity": "hard_blocker_for_direct_video_representation_claims",
    "applies_to": ["Parallax", "HiNeRV", "HNeRV", "SNeRV"],
    "evidence": "arXiv and README define Parameterized Local Linear Attention for language modeling.",
    "required_resolution": "Do not treat Parallax as scene-motion/3D/video evidence; only translate specific attention/gating/optimizer mechanisms into Pact-native charged experiments."
  },
  {
    "blocker_id": "tilde_parallax_no_byte_closed_archive_path",
    "severity": "hard_blocker_for_promotion",
    "applies_to": ["Parallax", "Wall Attention"],
    "evidence": "Repos expose PyTorch/Triton/CUDA kernels and LLM training scripts, with no Pact archive, inflate, receiver proof, or scorer replay.",
    "required_resolution": "Any Pact experiment must build a byte-charged archive/runtime path and remain false-authority until receiver replay and exact contest axis evidence exist."
  },
  {
    "blocker_id": "tilde_runtime_dependency_burden",
    "severity": "hard_blocker_for_direct_runtime_import",
    "applies_to": ["Parallax", "Wall Attention", "FlashLLA"],
    "evidence": "Parallax depends on torch 2.9.1/triton 3.5.1 and optional nvidia-cutlass-dsl Hopper decode; Wall depends on torch/triton/einops/flash-linear-attention.",
    "required_resolution": "Do not import these kernels into Pact inflate/runtime; reimplement only tiny charged primitives if a smoke justifies it."
  },
  {
    "blocker_id": "aurora_not_pr95_source_authority",
    "severity": "blocker_for_uncontrolled_optimizer_swap",
    "applies_to": ["PR95", "HNeRV", "HiNeRV"],
    "evidence": "Aurora is Muon-adjacent Tilde optimizer code, while PR95 control authority comes from the existing PR95/HNeRV curriculum and final Muon stage.",
    "required_resolution": "Compare Aurora against the existing Pact Muon/PR95 optimizer stage in a narrow advisory timing/convergence smoke before queue or dispatch admission."
  },
  {
    "blocker_id": "snerv_temporal_gate_requires_official_and_byte_charged_path",
    "severity": "blocker_for_scorer_or_rate_claim",
    "applies_to": ["SNeRV"],
    "evidence": "Current Pact SNeRV blockers still include official MFU/HFR/TUB parity and LF-byte dominance.",
    "required_resolution": "Close or explicitly bypass with a false-authority side-smoke label; charge all learned gate bytes and prove receiver replay before any rate claim."
  }
]
```

## Operator-Routable Queue: Top 3 Next Actions

1. `immediate_code_slice`: add an `aurora`/`aurora_like` optimizer-control row only in the NeRV plan/contract layer, default off and false-authority, reusing the existing optimizer-control schema rather than changing trainers first.
   - Target files/tools: `src/tac/analysis/nerv_long_training_campaign_plan.py`, `src/tac/analysis/nerv_candidate_feedback.py`, existing `SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS` surface under `src/tac/substrates/_shared/mlx_score_aware/`.
   - Done criterion: plan renders Aurora as `timing_smoke_candidate`, blocked from dispatch until a local timing/convergence artifact exists; no score authority.

2. `timing_smoke_candidate`: after the optimizer row exists, run a tiny local advisory PR95/HiNeRV optimizer smoke comparing existing `pact_muon_adamw`/`muon` versus Aurora-like row on an already-small candidate.
   - Target files/tools: existing NeRV long-training campaign planner/runner path; output under SSD if any nontrivial artifact is produced.
   - Required evidence: seconds/epoch, loss/component telemetry, byte ceiling unchanged, false-authority flags, no exact-axis claim.

3. `source_parity_input`: preserve Wall Attention as a SNeRV LF temporal-gate design input, but do not code it before `SNERV_OFFICIAL_MFU_HFR_TUB_PARITY_PROOF` unless explicitly labelled as a side smoke.
   - Target files/tools when ready: `src/tac/substrates/snerv_inverse_steg_carrier/official_tub.py`, `src/tac/analysis/snerv_lf_payload_archive_recode.py`, `src/tac/analysis/nerv_long_training_campaign_plan.py`.
   - Done criterion: machine-readable work order that charges learned gate bytes and reports LF-byte delta, receiver replay status, SegNet/PoseNet deltas, and blockers.

## Bottom Line

Tilde Research is real and active, but Parallax is not a visual-parallax codec lead, and the public Parallax repos are `Yifei-Zuo/*` rather than `tilde-research/*`. The actionable Pact signal is narrow: Aurora is a plausible Muon-adjacent optimizer smoke for PR95/HiNeRV/HNeRV, and Wall Attention is a primary-source idea input for a future byte-charged SNeRV LF temporal gate. Direct Parallax/Wall kernel imports are not contest-runtime faithful and should be rejected unless rebuilt as tiny Pact-native charged primitives.
