# Codex findings: Boosting-NeRV / BoostNeRV research integration

UTC: 2026-05-20T19:14:25Z
Agent posture: adversarial research/integration review only.
Score claim: false.
Promotion claim: false.
Exact local evidence: absent.
PR110 live submission files touched: no.
Code edits in this pass: none.

## Executive verdict

Boosting-NeRV is directly relevant to HNeRV/NeRV-family video compression, but
the primary-source mechanism is not the iterative residual-chain mechanism in
the local `boost_nerv` scaffold. The supported primary source is Zhang et al.,
CVPR 2024, "Boosting Neural Representations for Videos with a Conditional
Decoder" (`arXiv:2402.18152`, official repo `Xinjie-Q/Boosting-NeRV`).

Primary Boosting-NeRV is best treated as a conditional-decoder/TAT/SFT adapter
for NeRV, E-NeRV, and HNeRV. For Pact, the highest-EV route is a small
source-faithful Boosting-HNeRV adapter over the existing PR95/HNeRV lineage or
the current Pact-NeRV design lane, not a PR110 byte-level patch and not the
current local residual-head scaffold unless that scaffold is renamed as a
separate Pact hypothesis.

## Provenance and source map

Primary paper and code:

- CVPR OpenAccess paper:
  `https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Boosting_Neural_Representations_for_Videos_with_a_Conditional_Decoder_CVPR_2024_paper.html`
- CVPR PDF:
  `https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Boosting_Neural_Representations_for_Videos_with_a_Conditional_Decoder_CVPR_2024_paper.pdf`
- CVPR supplemental:
  `https://openaccess.thecvf.com/content/CVPR2024/supplemental/Zhang_Boosting_Neural_Representations_CVPR_2024_supplemental.pdf`
- arXiv:
  `https://arxiv.org/abs/2402.18152`
- Official repository:
  `https://github.com/Xinjie-Q/Boosting-NeRV`
- Official repo local forensic clone:
  `/tmp/boosting-nerv-official`, HEAD
  `d59ca91e7bae284a8970db007e5b2c7f804b0b46`, last commit
  `2024-04-19T10:34:23+08:00`, license Apache-2.0.

Secondary/index sources checked:

- OpenAlex work record: `https://openalex.org/W4392340572`.
- OpenAlex reported `cited_by_count=2` for this arXiv work in the queried
  record. Treat this as an index-specific lower bound, not a global citation
  count.
- OpenAlex citing works from this query:
  `PNVC: Towards Practical INR-based Video Compression`
  (`https://doi.org/10.1609/aaai.v39i3.32315`, AAAI 2025) and
  `Fanerv: Frequency Separation and Augmentation Based Neural Representation
  for Video` (`https://doi.org/10.2139/ssrn.5271789`, SSRN 2025).

Negative source finding:

- I found no primary paper/repo evidence for the local anchor
  `Liu et al. ECCV 2024 "BoostNeRV: Iterative Refinement for Implicit Neural
  Video Representations"` or a real `arXiv:2407.xxxxx` BoostNeRV paper matching
  that description. The current local anchor should be treated as false or
  at least unsupported until an operator provides a missing primary source.

## Local references found

Tracked BoostNeRV references:

- `git ls-files | rg -i 'boost[-_ ]?nerv|boostnerv|boost_nerv'` found no
  tracked BoostNeRV implementation files in this checkout.

Untracked/local WIP references:

- `src/tac/substrates/boost_nerv/__init__.py`
- `src/tac/substrates/boost_nerv/architecture.py`
- `src/tac/substrates/boost_nerv/archive.py`
- `src/tac/substrates/boost_nerv/inflate.py`
- `src/tac/substrates/boost_nerv/score_aware_loss.py`
- `src/tac/substrates/boost_nerv/tests/test_boost_nerv.py`
- `experiments/train_substrate_boost_nerv.py`
- `scripts/remote_lane_substrate_boost_nerv.sh`
- `.omx/operator_authorize_recipes/substrate_boost_nerv_modal_t4_dispatch.yaml`
- `.omx/research/boost_nerv_l0_scaffold_design_20260520T184500Z.md`

State references:

- `.omx/state/lane_registry.json` has
  `lane_boost_nerv_l0_scaffold_20260520`, phase `2.0`, level `1`, with
  `real_archive_empirical=false`, `contest_cuda=false`, `contest_cpu=false`,
  and notes describing an iterative boosting residual chain with `BSV1` magic.
- `.omx/state/subagent_progress.jsonl` records active/in-progress ownership by
  `wave-3-nerv-literature-l0-rescoped-boostnerv-nirvana-coinpp-20260520`,
  including `src/tac/substrates/boost_nerv/*`,
  `experiments/train_substrate_boost_nerv.py`, recipe, and remote driver.

Related local boosting namespace memos:

- `.omx/research/tac_boosting_namespace_design_20260517.md`
- `.omx/research/tac_boosting_parallel_merge_runtime_hardening_20260517_codex.md`

Those memos are useful for generic composition discipline, but they are not a
source-faithful Boosting-NeRV implementation.

## Paper/repo status

Paper identity:

- Title: "Boosting Neural Representations for Videos with a Conditional
  Decoder".
- Authors: Xinjie Zhang, Ren Yang, Dailan He, Xingtong Ge, Tongda Xu,
  Yan Wang, Hongwei Qin, Jun Zhang.
- Venue/status: CVPR 2024; official repo labels it CVPR 2024 Highlight.
- arXiv: `2402.18152`, submitted 2024-02-28, v3 2024-03-16.

Official code status:

- Repo: `Xinjie-Q/Boosting-NeRV`, Apache-2.0.
- README says code was released 2024-04-18.
- Official implementation includes boosted NeRV, E-NeRV, and HNeRV model
  variants plus representation, compression, inpainting, and interpolation
  scripts.
- Compression path follows the paper: overfit a per-video representation, then
  apply consistent entropy minimization (CEM) and quantization.
- Requirements include old/heavy dependencies:
  `torch==1.8.1`, `torchvision==0.9.1`, `timm==0.6.7`, `decord==0.6.0`,
  `compressai==1.2.0`, `constriction==0.3.1`, `pytorchvideo==0.1.5`,
  `pytorch_msssim==0.2.1`, and related packages.

## Algorithmic summary

Boosting-NeRV is a conditional decoder framework for implicit neural video
representations. The key idea is to improve alignment of intermediate decoder
features to the target frame by conditioning decoder blocks on frame index.

Primary mechanisms:

- Temporal-aware affine transform (TAT): uses frame index as a prior condition.
- SFT-style modulation in code: `SFTLayer` predicts per-feature `scale` and
  `shift`, then applies `feature * (scale + 1) + shift`.
- Conditional decoder blocks: official `NeRVBlock.forward` accepts
  `(feature, embed)` and applies an SFT residual block when enabled.
- HNeRV-Boost: keeps an image/frame embedding path, builds `t_embed` from a
  positional encoding of normalized frame index through `stem_t`, then applies
  that same temporal condition through the decoder blocks.
- Sinusoidal NeRV-like block: introduced to improve intermediate feature
  diversity and parameter distribution.
- High-frequency preserving reconstruction loss: used during representation
  fitting.
- Compression: official path uses quantizers and CEM/entropy modeling; repo
  code uses `constriction`/ANS and CompressAI-related entropy utilities for
  bit estimation/coding.

Important non-equivalence:

- The primary Boosting-NeRV paper is not a gradient-boosted residual cascade
  with multiple residual heads. The local `boost_nerv` residual-chain scaffold
  may be a valid Pact hypothesis, but it is not source-faithful Boosting-NeRV
  under the primary sources found in this pass.

## Implementation feasibility for Pact

Feasible, but the source-faithful integration should be narrower than the
official repo:

- Implement or prototype a `Boosting-HNeRV-TAT` adapter over an existing Pact
  HNeRV/PR95/Pact-NeRV base, adding a small temporal-conditioning branch and
  SFT/TAT modulation at decoder blocks.
- Keep the adapter export-first: monolithic `0.bin`, deterministic metadata,
  quantized weights/latents, no scorer loads at inflate time, and no network or
  package installation in inflate.
- Use score-aware/eval_roundtrip training in Pact rather than UVG PSNR-only
  losses. The official paper's PSNR/MS-SSIM gains do not directly imply
  SegNet/PoseNet movement.
- Treat official entropy coding as design evidence, not a drop-in runtime. A
  contest-safe implementation should use existing Pact archive/packer patterns
  or a tiny vendored entropy decoder with explicit custody and dependency
  closure.
- The local residual-head scaffold should be corrected before any dispatch:
  either rename it as `boost_residual_nerv` / Pact-only hypothesis, or replace
  its architecture/literature metadata with a source-faithful TAT/SFT adapter.

I did not add a new planning scaffold in this pass because the BoostNeRV files
are actively owned by a sibling subagent and a memo-only correction is the
lowest-conflict integration artifact.

## Compliance and runtime risks

- Dependency risk: official requirements are old and heavy. They cannot be
  assumed contest-safe for inflate or exact eval without a fresh runtime closure
  audit.
- Inflate risk: `decord`, `timm`, `compressai`, `constriction`, dataset loaders,
  training scripts, or scorer imports must not leak into the final inflate path.
- Runtime cost risk: the paper reports boosted variants add decoder latency
  relative to baselines. This is acceptable only if exact contest runtime stays
  inside limits and the byte/distortion tradeoff moves the contest objective.
- License/custody risk: Apache-2.0 code can be reused with notice, but any
  vendored source must preserve attribution and runtime-tree custody.
- Axis risk: UVG PSNR/MS-SSIM and V100 decoding timings are not contest CUDA/T4
  SegNet/PoseNet evidence. No ranking, promotion, or retirement should use those
  source metrics as equivalent to Pact exact eval.
- Local metadata risk: current local WIP repeatedly cites unsupported
  `Liu ECCV 2024` and implements a different residual-chain mechanism. That is
  a pre-dispatch correctness bug, not just a citation typo.

## Stack position versus current Pact lanes

FEC6 / PR101 / PR110:

- Boosting-NeRV is not a byte-only same-runtime patch for FEC6/PR110. It is a
  representation substrate or a decoder adapter.
- FEC6/PR110 are rate-heavy, so a full representation replacement can matter,
  but only after a byte-closed archive/runtime packet exists.
- Do not edit PR110 live submission files for this lane.

HNeRV / PR95:

- Highest direct fit. The official paper explicitly boosts HNeRV, and the code
  contains `HNeRV_Boost`.
- Best first source-faithful path is HNeRV + temporal conditioning/SFT at
  decoder blocks, with score-aware Pact training and export custody.

Selector / q-byte / signed waterbucket:

- These are downstream or orthogonal byte-layout tools. Once a Boosting-HNeRV
  packet exists, selector/q-byte/waterbucket can choose modes, tune quantized
  weights/latents, or allocate bytes to TAT/SFT parameters.
- They are not evidence that the representation itself improves scorer
  distortion.

RAFT / ego-motion / foveation / LA-pose:

- These are plausible teacher or allocation priors for conditioning, hard-pair
  schedules, or region-weighted loss.
- They should not become runtime dependencies unless the archive/runtime packet
  explicitly proves contest-safe closure.

Diffusion / generative lanes:

- Useful mainly as compress-time teachers or priors. Runtime diffusion is too
  heavy for this contest path unless a separate exact-runtime miracle is proven.

Theoretical-floor work:

- Relevant because the official method targets the same family of
  function-as-video representations and could reduce distortion at modest added
  decoder parameters.
- It does not change the mathematical floor by itself; it supplies an
  implementation hypothesis that must compete against charged bytes and exact
  SegNet/PoseNet loss.

## Speculative score-movement predictions

All predictions in this section are speculative and are not score claims.

- [speculative] A source-faithful TAT/SFT adapter over HNeRV is more likely to
  produce useful contest signal than the current local residual-head scaffold,
  because it is the mechanism actually validated by the primary paper and it
  directly targets intermediate-feature/frame alignment.
- [speculative] The strongest upside is hard-pair distortion movement at a
  similar byte budget if temporal conditioning fixes frames/pairs where a
  static decoder underfits high-frequency or pose/seg-sensitive structure.
- [speculative] The main failure mode is rate/runtime tax: extra conditioning
  parameters and slower decoder blocks can erase distortion gains under the
  contest objective.
- [speculative] Official-size UVG HNeRV-Boost models are probably too large as
  direct Pact packets. A Pact version needs aggressive downsizing, quantization,
  and archive-aware training from the start.
- [speculative] The first useful gate is not a full dispatch. It is a tiny
  source-faithful TAT/SFT HNeRV smoke that measures whether score-aware local
  advisory loss moves on the hard pairs at fixed or explicitly measured bytes.

## Recommended next action

Do not dispatch or promote the current local `boost_nerv` scaffold as
Boosting-NeRV. First, make a small source-faithfulness correction pass after the
active WIP owner is clear:

1. Correct all local metadata from the unsupported `Liu ECCV 2024` anchor to
   Zhang et al. CVPR 2024 `arXiv:2402.18152`.
2. Decide explicitly whether the current residual-head implementation is a
   Pact-only hypothesis. If yes, rename/tag it away from source-faithful
   Boosting-NeRV. If no, replace it with a TAT/SFT conditional decoder adapter.
3. Build the first integration artifact as `Boosting-HNeRV-TAT` over the
   existing PR95/HNeRV or Pact-NeRV path, with `eval_roundtrip=True`,
   export-first archive grammar, no scorer loads at inflate time, and a tiny
   CPU/local advisory smoke before any paid exact-CUDA work.

Files changed by this pass:

- `.omx/research/codex_findings_boostnerv_research_integration_20260520T191425Z_codex.md`

Tests run:

- None. Memo-only research/integration landing; no code changed.
