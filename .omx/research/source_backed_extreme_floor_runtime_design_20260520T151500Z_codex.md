# Source-Backed Extreme-Floor Runtime Design

**UTC:** 2026-05-20T15:15:00Z
**Owner:** Codex
**Lane:** `lane_theoretical_floor_source_backed_candidates_20260520`
**Authority:** implementation/runtime routing only; `score_claim=false`;
`promotion_eligible=false`; `ready_for_exact_eval_dispatch=false`.

## Correction To The Floor Premise

The quoted "segmentation term alone contributes ~0.18" premise conflates raw
distortion with weighted score terms. The verified PR #110 CPU decomposition
is:

- weighted SegNet term: `0.056029`
- weighted PoseNet term: `0.017155`
- weighted rate term: `0.118867`
- distortion-only zero-archive score: `0.073184`
- rate-only perfect-distortion score: `0.118867`
- recomputed total: `0.192051`

This changes the optimization target. PR #110 is rate-heavy, not
segmentation-term-limited at `0.18`. The aggressive path to the floor must
attack both sides:

1. reduce or replace the `178517` charged bytes;
2. keep or improve the `0.073184` distortion sum while doing so;
3. only use large external models as compress-time teachers unless their
   weights and runtime are explicitly charged in the archive.

## Source Interpretation

Primary sources refreshed:

- LA-Pose (`https://arxiv.org/abs/2604.27448`): inverse/forward-dynamics
  latent action features repurposed as camera-pose estimation inputs. Pact use:
  compress-time pose/motion teacher, not inflate dependency.
- Telescope (`https://princeton-computational-imaging.github.io/Telescope/`):
  learnable hyperbolic foveation magnifies distant regions and compresses
  nearby regions for long-range perception. Pact use: compact charged
  foveation geometry/field, not the full detector.
- SEA-RAFT (`https://arxiv.org/abs/2405.14793`): efficient optical-flow teacher
  with faster RAFT-style convergence. Pact use: compress-time motion/FOE
  teacher for foveation and pose atoms.
- Thinking with Visual Primitives
  (`https://www.k-a.in/Thinking_with_Visual_Primitives.pdf`): points/boxes as
  sparse visual primitives. Pact use: small charged point/box/region hints for
  foveation or bit allocation, not an MLLM runtime.

## Runtime Contracts Implemented

Added:

- `src/tac/theoretical_floor_candidates.py`
- `tools/build_theoretical_floor_candidate_matrix.py`
- `src/tac/tests/test_theoretical_floor_candidates.py`

Emitted:

- `.omx/research/theoretical_floor_candidate_matrix_20260520_codex/candidate_matrix.json`
- `.omx/research/theoretical_floor_candidate_matrix_20260520_codex/candidate_matrix.md`

The matrix pins four source-backed runtime contracts:

| candidate | runtime design | first gate | authority |
|---|---|---|---|
| `tf_siren_first_anchor` | full SIREN/INR renderer; runtime consumes `0.bin` and small torch decoder | CPU smoke | local smoke only |
| `tf_telescope_lfv1_pose_foveation` | charged LFV1 tuple payload lowered to HFV1 foveation params plus runtime consumer | byte-closed local archive | fail-closed archive custody |
| `tf_vqvae_full_renderer` | charged VQ codebook, decoder, token stream | focused VQ tests/export gate | local readiness only |
| `tf_c3_coolchic_sparse_residual` | sparse learned-codec residual sidecar on proven base runtime | `l2_encoded` materializer with explicit raw inputs and byte budget | research signal only |

## Artifact Pushed

Built a source-backed LFV1/Telescope-pose foveation archive:

```text
experiments/results/theoretical_floor_lfv1_pose_foveation_20260520_codex/archive_candidate/archive.zip
```

Facts:

- archive bytes: `68460`
- archive SHA-256: `6a353413f67e75d4db94839572573e86d3ca811e64bba0137d3e9c65a09dec7c`
- LFV1 payload bytes: `402`
- LFV1 payload SHA-256: `be4d51599b715196c78cf5e2824290990e9fb1f56c835dfa4c7ea12365167e41`
- `foveation_params.bin` bytes: `23696`
- `foveation_params.bin` SHA-256: `e4c10b98f8e686d9534451c21b5126bce2115073c0b68d8d2df0d1309e89a275`

The candidate is intentionally fail-closed:

- `score_claim=false`
- `dispatch_attempted=false`
- `ready_for_exact_eval_dispatch=false`

Readiness blockers:

- `runtime_loader_parity_not_passed`
- `lapose_foveation_scorer_visible_output_parity_not_proven`
- `lapose_foveation_runtime_output_parity_not_proven`
- `exact_cuda_auth_eval_missing`

This is progress because the source stack is now deterministic archive bytes
with measured member custody, not only a roadmap claim.

## SIREN Local Smoke

Ran the first source-backed SIREN local smoke:

```bash
.venv/bin/python experiments/train_substrate_siren.py \
  --video-path upstream/videos/0.mkv \
  --output-dir experiments/results/siren_source_backed_floor_smoke_20260520_codex \
  --epochs 3 \
  --device cpu --smoke \
  --skip-archive-build --skip-auth-eval
```

Result:

- params: `2438`
- losses: `1.0080 -> 1.0039 -> 0.9999`
- checkpoint:
  `experiments/results/siren_source_backed_floor_smoke_20260520_codex/smoke_checkpoint.pt`
- checkpoint SHA-256:
  `81df5f88dca427c300da79a492e46de0f46e7e4e8ed6360aaf5ab9986b4bd301`

This is not score evidence. It verifies that the current SIREN training surface
is live after the readiness-gate patch.

## Full-Stack Optimization Stack

The highest-EV stack is now:

1. **Teacher pass, compress-time only:** SEA-RAFT/LA-Pose/visual primitives
   produce flow, FOE, horizon, hard-pair, point/box, and pose-confidence
   records.
2. **Primitive compiler:** records become compact atoms: LFV1 foveation tuples,
   point/box region hints, selector priors, or codebook priors.
3. **Representation candidates:** SIREN full renderer and VQ-VAE full renderer
   attack the rate-heavy archive term by replacing HNeRV bytes.
4. **Residual candidates:** C3/Cool-Chic sparse residuals and LFV1 foveation
   operate on top of proven base decodes where a full replacement is not yet
   competitive.
5. **Deterministic byte correction:** master-gradient / pair-gradient nudges
   are applied only after a byte-closed candidate exists, to avoid optimizing
   an unshipped abstraction.
6. **Exact-eval gate:** no score or promotion until archive/runtime custody,
   no-op controls, component recompute, and exact CPU/CUDA axis evidence exist.

## Next Implementation Step

Do not spend GPU yet from this memo. The next local implementation should
replace the LFV1 fail-closed `runtime_consumer.py` skeleton with a scorer-visible
runtime bridge around a real base decode:

1. choose PR106 or PR110 as base runtime;
2. consume `foveation_params.bin` in the runtime;
3. apply the hyperbolic/foveation warp to actual decoded RGB outputs;
4. add identity-vs-mutated-payload no-op controls over full frames;
5. only then consider claimed exact eval.

This is the shortest path from the new papers to a contest-relevant runtime
artifact.
