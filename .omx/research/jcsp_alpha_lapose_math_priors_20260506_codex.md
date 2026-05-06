# JCSP / Alpha / LA-Pose Math-Prior Contract Ledger (2026-05-06)

Scope: paper-derived priors only, mapped to hardened local contracts. No GPU
dispatch, no score claim, no promotion claim.

## Contract Mapping

| Source | Exact mathematical variable / constraint extracted | Scorer term targeted | Bytes charged | Local implementation surface | Evidence grade | Dispatch blockers | Fail-closed criteria |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LA-Pose project page, "latent action features as inputs to a camera pose estimator" and relative pose head for translation/rotation/FOV/scale: https://la-pose.github.io/ ; arXiv:2604.27448 https://arxiv.org/abs/2604.27448 | Treat compress-time motion prior as an init tensor `P_init in R^(N_pairs x 6)` only. Constraint: `shape(P_init) = (N, 6)`, finite fp32, no LA-Pose model or sidecar at inflate time. | pose | `N_pairs * 6 * dtype_bytes` only after archive rebuild; current `--init-poses` is training/planning input, charged bytes unclaimed. | `experiments/optimize_poses.py --init-poses`; `_load_init_pose_tensor`; tests in `src/tac/tests/test_optimize_poses_init_poses.py`. | external + local contract | exact archive rebuild missing; CUDA auth eval missing; dispatch claim missing; init-source manifest missing. | Refuse if tensor missing, wrong shape, too few rows, NaN/Inf, or ambiguous with `--seed-poses-path` / `--gt-poses-path`. |
| RAFT paper/repo, all-pairs correlation volume and recurrent flow update: https://arxiv.org/abs/2003.12039 ; https://github.com/princeton-vl/RAFT | Treat optical flow as side information used to derive `P_init`; no RAFT weights or flow sidecar may be consumed at inflate time. Local variable is still `P_init`, not flow. | pose | Same as `P_init` row; RAFT flow bytes are not charged because they are not an archive member in this slice. | Existing `experiments/derive_poses_from_raft.py` plus new `--init-poses` consumer in `experiments/optimize_poses.py`. | external + empirical non-score local manifest | RAFT manifest must exist; exact archive rebuild missing; CUDA auth eval missing. | Refuse dispatch if only flow exists without a validated `(N, 6)` pose tensor and custody manifest. |
| Geisler and Perry foveated multiresolution pyramid: https://svi.cps.utexas.edu/spie1998.pdf | Use eccentricity/level idea as a stream-priority prior, not a score. Variables: level `l`, eccentricity `e`, critical radius `e_c`; local translation is per-stream marginal `dScore/dByte` plus charged bytes. | seg + rate | Per stream `bytes_charged = byte_estimate` until exact encoded bytes exist. | `JCSPTensorStreamSpec.scorer_term_targeted`, `score_per_byte_marginal`, `bytes_charged` in `src/tac/joint_codec_stack_orchestrator.py`; tests in `test_jcsp_model_streams.py`. | derivation + local contract | qint/wire stream missing; decode validation missing; score-marginal artifact missing. | Refuse dispatch if marginal, decode validation, or charged bytes are missing. |
| Chang, Mallat, Yap wavelet foveation: https://www.sciencedirect.com/science/article/pii/S1063520300903245 | Space-variant smoothing kernel is diagonal-dominant in wavelet bases; local constraint is that any wavelet mask atom must inverse-decode to declared mask tensor before training reuse. | seg + rate | `bytes_charged` required by readiness manifest; absent bytes block readiness. | `src/tac/alpha_mask_codec_readiness.py` family `wavelet`; canonical decoder `tac.wavelet_mask_codec.decode_wavelet_codec`. | derivation + local contract | decode validation missing; charged bytes missing; exact archive/eval missing. | Refuse if decoder mismatch, decoded shape/dtype/hash missing, class range invalid, sidecars required, or bytes absent. |
| Hua and Liu dual-sensor foveated imaging: https://opg.optica.org/ao/abstract.cfm?uri=ao-47-3-317 | Limited pixels/bandwidth are allocated as a function of foveal eccentricity; local constraint is explicit stream allocation rather than implicit uniform treatment. | seg + rate | Per stream only; no global score claim. | `model_to_jcsp_streams(model)` emits deterministic tensor streams with byte and marginal annotations. | derivation + local contract | same as JCSP row. | same as JCSP row. |
| Filler, Judas, Fridrich STC additive distortion: https://ws2.binghamton.edu/fridrich/Research/stc-v7-double-column.pdf | Additive distortion `D(x,y) = sum_i rho_i(x_i,y_i)` maps locally to per-stream additive marginal `m_s = dScore/dByte`; missing `m_s` is a blocker, not zero evidence. | joint, with explicit `scorer_term_targeted` | `bytes_charged` per tensor stream; exact encoded bytes required before dispatch. | `model_to_jcsp_streams`, `JCSPTensorStreamSpec.constraint_tags`, dispatch blockers. | derivation + local contract | score-marginal artifact missing; qint/wire stream missing; decode validation missing. | Refuse if any stream lacks cached marginal evidence or charged-byte/decode closure. |
| Fridrich, Goljan, Soukal wet paper codes with improved embedding efficiency: https://ws2.binghamton.edu/fridrich/Research/wpc_with_improved_embedding_efficiency-ieee.pdf | Selection channel may be non-shared; local variable is `wet_streams`, an explicit set/prefix of streams that must not be perturbed without override. | joint safety constraint | Wet streams keep raw/estimated bytes; no savings claim. | `model_to_jcsp_streams(..., wet_streams=...)` adds `wet_stream_do_not_perturb` and dispatch blocker. | derivation + local contract | wet stream requires explicit override; exact archive/eval missing. | Refuse if a wet stream would be dispatched without explicit override and validation. |
| Fridrich, Goljan, Soukal perturbed quantization: https://ws2.binghamton.edu/fridrich/Research/p44-fridrich.pdf | Side-informed quantization uses pre-processing side information but recipient must not need it. Local constraint: VQ-VAE/grayscale/RAFT side info is compress-time only; decode must close from archive bytes alone. | seg for masks, pose for init poses | Mask codec `bytes_charged` required in readiness manifest; pose init bytes charged only after archive rebuild. | `alpha_mask_codec_readiness.py` families `vqvae`, `grayscale_lut`; `optimize_poses.py --init-poses`. | derivation + local contract | sidecars required; decode validation missing; exact archive/eval missing. | Refuse if decode validation reports `sidecars_required=true`, missing decoder entrypoint, missing hashes, or ambiguous init source. |

## Local Artifacts Created

- `src/tac/joint_codec_stack_orchestrator.py::model_to_jcsp_streams`
- `src/tac/alpha_mask_codec_readiness.py::build_alpha_mask_training_readiness_contract`
- `experiments/optimize_poses.py --init-poses`
- Tests:
  - `src/tac/tests/test_jcsp_model_streams.py`
  - `src/tac/tests/test_alpha_mask_codec_readiness.py`
  - `src/tac/tests/test_optimize_poses_init_poses.py`

## Non-Claims

- No exact score.
- No CUDA auth eval.
- No GPU dispatch.
- No claim that LA-Pose, RAFT, foveation, STC, wet-paper coding, or perturbed
  quantization improves the contest score until an exact archive passes the
  repository dispatch and auth-eval gates.

## Integration Update

JCSP tensor stream specs and alpha mask readiness manifests now carry
`research_basis_ids` directly. Downstream field-equation and meta-Lagrangian
tools can therefore preserve paper provenance without scraping this ledger.
