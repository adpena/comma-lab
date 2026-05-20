# Zipper Source Map Surface Audit

**UTC:** 2026-05-20T14:55:23Z  
**Owner:** Codex  
**Source package:** `.omx/research/inbox_zipper_20260520T144021Z_codex/research_package.zip`  
**Source package SHA-256:** `9ffce5d7802ebdb0669b21888b7ec76496561dee652c72f7993be803c38d2506`  
**Lane:** `lane_zipper_package_intake_20260520`

## Verdict

Zipper is useful as a roadmap and reviewer model, but it is not implementation
authority. Most frontier ideas named in the package already have stronger
repo-native surfaces than the included pseudocode. The correct integration
path is therefore:

1. keep the PR #110 public surface focused and already-applied;
2. preserve the FEC7 byte-only result as a terminal local negative for current
   FEC6 selector bytes;
3. route the remaining ideas through existing `tac` modules, readiness gates,
   packet compilers, and lane-claim discipline;
4. reject package commands that reference nonexistent scripts or bypass Pact
   provider/runtime contracts.

The package scaffolds under `IMPLEMENTATION_SCAFFOLD/` should not be copied
into `src/tac`. They are pseudocode, and in several cases they would regress
existing archive-grammar, export, or score-authority discipline.

## Source Map Disposition Table

| Zipper lane | Package claim | Repo-native surfaces found | Classification | Follow-up gate |
|---|---|---|---|---|
| PR #110 public packet | Tighten PR body, dependency wording, CPU/CUDA evidence, saturation wording, and maintainer-facing tone. | Live PR #110 body; GitHub release body/title; `.omx/research/pr110_live_surface_update_applied_20260520T143154Z_codex.md`. | `LANDED_TOOLING` / `APPLIED_PUBLIC_SURFACE` | No more live edits unless maintainer feedback or fact drift requires them. Do not push branch commits or replace release asset without explicit operator approval. |
| FEC7 selector | 63-mode / 32-active selector and fixed Huffman recode. | `tools/profile_pr101_fec7_selector_entropy.py`; `src/tac/packet_compiler/pr101_fec7_selector.py`; `src/tac/tests/test_pr101_fec7_selector_entropy.py`; `experiments/results/pr110_zipper_fec7_selector_profile_20260520_codex/profile.json`. | `TERMINAL_LOCAL_NEGATIVE` for byte-only recoding on current FEC6 bytes. | Reactivate only with a new charged selector model whose model+stream saves at least 79 bytes versus FEC6, or with a component-score-improving selector. Current best charged FEC7 candidate is 268 bytes vs 249 bytes FEC6 selector. |
| Deterministic packet compiler | Build a packet compiler and manifest generator before future exports. | `src/tac/packet_compiler/deterministic_compiler.py`; `src/tac/packet_compiler/deterministic_compiler_cli.py`; `tools/contest_packet_compiler.py`; `tools/submission_packet_compiler.py`; `src/tac/submission_packet_compiler.py`; packet-compiler golden vectors and tests. | `LANDED_TOOLING` | Future candidate archives should use existing compiler surfaces. Do not create a parallel `tools/packet_compiler.py` from package prose. |
| SIREN / INR | Train a SIREN INR codec and export weights into an archive. | `src/tac/residual_basis/siren_residual.py`; `experiments/train_substrate_siren.py`; `tools/audit_siren_substrate_readiness.py`; `tools/materialize_siren_residual_pr106_sidecar.py`; `src/tac/tests/test_siren_substrate_readiness.py`; `.omx/research/staged_siren_residual_pr106_sidecar_ready_to_dispatch_20260511T174843Z.md`. | `SCAFFOLDED_WITH_READINESS_GATE` | Run the SIREN readiness audit. Promotion remains blocked until a hermetic inflate grammar, byte-closed archive, and exact-eval axis evidence exist. |
| VQ-VAE full renderer | Discrete latent codec with entropy-coded tokens. | `src/tac/vqvae_as_full_renderer.py`; `experiments/train_vqvae_as_renderer.py`; `src/tac/vqvae_mask_codec.py`; `src/tac/contrib/vqvae_codec.py`; `src/tac/tests/test_train_vqvae_as_renderer.py`; `src/tac/tests/test_vqvae_mask_codec.py`. | `SCAFFOLDED_WITH_ARCHIVE_GRAMMAR` | Run focused VQ tests and record blockers. Existing module has an archive grammar blueprint and codebook-collapse gate; package pseudocode is weaker than this surface. |
| Cool-Chic / C3 | Learned neural codec families may beat HNeRV. | `src/tac/contrib/coolchic_renderer.py`; `src/tac/residual_basis/cool_chic_residual.py`; `src/tac/residual_basis/c3_residual.py`; `src/tac/residual_basis/c3_encoder_l2.py`; `tools/materialize_cool_chic_residual_pr106_sidecar.py`; `tools/materialize_c3_residual_pr106_sidecar.py`; `experiments/train_substrate_cool_chic.py`; `src/tac/tests/test_coolchic_darts.py`. | `SCAFFOLDED_WITH_EXPORT_GATE` | Produce a no-spend smoke/runbook from existing materializers. No GPU dispatch or score claim from package budgets. |
| Foveation / LA-Pose | Allocate more bits to task-sensitive spatial regions and pose geometry. | `src/tac/foveation_field.py`; `src/tac/lapose_foveation_atoms.py`; `src/tac/lapose_foveation_runtime_skeleton.py`; `src/tac/analysis/lapose_foveation_payload.py`; `tools/build_lapose_foveation_payload_archive.py`; `tools/probe_c1_foveation_vs_uniform_quantization_disambiguator.py`; `tools/audit_hyperbolic_foveation_readiness.py`; foveation tests. | `SCAFFOLDED_WITH_PAYLOAD_GATE` | Run local payload/readiness tests before any dispatch. The repo already enforces byte budget, no-op detection, MPS rejection, and archive-format declarations. |
| RAFT / pose prior | Use optical flow / pose priors to reduce PoseNet distortion. | `src/tac/raft_pose.py`; `src/tac/raft_radial_pose.py`; `src/tac/raft_pose_stream.py`; `src/tac/codec_pipeline_raft_pose.py`; `experiments/derive_poses_from_raft.py`; `experiments/compute_raft_flow.py`; RAFT tests. | `SCAFFOLDED_CUDA_AWARE` | Keep CPU-only tests local; any real RAFT training/eval dispatch needs lane claim and provider runtime probe. |
| UNIWARD / STC / Fridrich family | Texture-adaptive bit allocation and syndrome-trellis coding. | `src/tac/uniward_delta.py`; `src/tac/uniward_texture.py`; `tools/build_uniward_stc_hessian_a1_v1.py`; `tools/pr101_arch_shrink_x_lagrangian_x_uniward_empirical.py`; `tools/pr101_omega_opt_uniward_weighted_allocation.py`; UNIWARD tests and exact-eval artifacts. | `SCAFFOLDED_WITH_PRIOR_NEGATIVE_ANCHORS` | Use master-gradient / scorer sensitivity, not the prior falsified mean-theta-squared route. Treat STC label skeptically until explicit STC implementation is present. |
| World-model priors | DreamerV3 / V-JEPA / Mamba-style predictive coding. | `tools/probe_c1_world_model_vs_independent_frames_disambiguator.py`; `tools/probe_c1_world_model_v2_posterior_prior_disambiguator.py`; `experiments/train_substrate_z5_predictive_coding_world_model.py`; `experiments/train_substrate_z7_mamba2*`; Z7 Mamba artifacts and memos. | `DESIGN_PLUS_PROXY_PROBE` | Continue only through existing probe and exact-export surfaces. No score/promotion from proxy world-model smokes. |
| MAE / VideoMAE / SAM / DINO / Falcon / SPADE / CLADE | External priors, dense features, semantic conditioning. | Mostly research memos and candidate inventory entries in this checkout; no reviewed byte-closed submission surface found in this pass. | `EXTERNAL_REFERENCE` / `DESIGN_ONLY` for Codex purposes. | Claude/research can refine designs; Codex should build gates only after a concrete design names runtime closure, archive grammar, and cheapest falsifying smoke. |
| Cloud GPU spend plan | Batch smokes across T4/4090/A100 with cost gates. | `tools/claim_lane_dispatch.py`; Modal/Lightning/Vast provider helpers; `.omx/state/active_lane_dispatch_claims.md`; provider runtime rules in `AGENTS.md`/`CLAUDE.md`. | `ADVISORY_ONLY` | Package spend plan does not authorize GPU. Any dispatch must pass lane claim, provider helper, runtime import probe, artifact custody, and terminal claim update. |
| Theoretical floor memo | Near-term 0.18-0.185 target; archive recompression saturated. | PR110 evidence, recompression sweeps, score formula, existing floor memos. | `ADVISORY_WITH_MATH_BUG` | Do not cite as floor authority until corrected. The current 178,517-byte archive term is `25 * 178517 / 37545489 ~= 0.118867`, not `0.0001-0.0002`; 259 bytes changes the score by about `0.00017245`. |

## Score Decomposition Correction

The Zipper floor discussion appears to conflate raw distortions with weighted
score terms. For PR #110 CPU evidence, `report.txt` gives:

- raw SegNet distortion: `0.00056029`, weighted score term
  `100 * 0.00056029 = 0.056029`;
- raw PoseNet distortion: `0.00002943`, weighted score term
  `sqrt(10 * 0.00002943) ~= 0.017155`;
- archive bytes: `178517`, weighted rate term
  `25 * 178517 / 37545489 ~= 0.118867`;
- total: `0.056029 + 0.017155 + 0.118867 ~= 0.192051`.

So the current distortion-only score at zero archive bytes would be about
`0.073184`, not `0.18`. The current rate-only score at perfect distortion
would be about `0.118867`. A phrase like "segmentation term alone contributes
~0.18" is only plausible for a hypothetical raw SegNet distortion near
`0.0018` after multiplying by the contest `100x` weight; it does not describe
the verified PR #110 component decomposition.

## Public Documentation Risk

`docs/asymptotic_floor_candidate_inventory.md` remains useful as a broad
research inventory, but it is not safe as a direct PR #110 dependency until it
is sanitized. It contains PR-body-like language that was intentionally removed
from the live PR, including stale broad claims about an HNeRV-family
`~0.0008` cluster and a loose description of CPU evaluation hardware. Keep the
live PR body linked only to root-level research repositories unless this doc is
updated and re-audited.

## Implementation Staircase From This Audit

1. **Complete:** PR #110 public-surface patching and release wording.
2. **Complete:** FEC7 byte-only selector recoding local terminal negative.
3. **Now:** SIREN readiness audit and VQ-VAE focused tests to convert
   scaffold status into explicit blockers or green local gates.
4. **Next:** Foveation/RAFT local gate: run existing payload/readiness tests or
   record missing-dependency blockers.
5. **Next:** Cool-Chic/C3 materializer smoke spec from existing sidecar tools;
   no dispatch from generic package commands.
6. **Only after local gates:** provider-spend dispatch with lane claim,
   runtime import probe, shipped-code manifest, exact archive custody, and
   terminal claim update.

## Non-Authority Flags

- `score_claim = false`
- `promotion_eligible = false`
- `ready_for_exact_eval_dispatch = false`
- `gpu_dispatch_authorized = false`
- `package_pseudocode_authoritative = false`


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:zipper-source-map-surface-audit-codex-memo-trigger-tokens-describe-audit-findings-not-new-equation -->
