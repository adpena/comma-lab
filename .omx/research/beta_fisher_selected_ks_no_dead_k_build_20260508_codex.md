# Beta-Fisher selected-K no-dead-K build - 2026-05-08

Scope: byte-closed CPU rebuild of the PR101 no-dead-K Path-B step-6 archive
using the beta-Fisher/Jacobian planning manifest's `selected_Ks`. This is a
builder/wiring artifact, not a score claim.

## Artifact

- Build dir:
  `experiments/results/admm_x_lossy_coarsening_path_b_step6_no_dead_k_20260508T092353Z/`
- Builder:
  `tools/build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py`
- Source planning manifest:
  `reports/raw/beta_fisher_lossy_coarsening_weights_20260508_codex_smoke/manifest.json`
- Archive bytes: `159,576`
- Archive SHA-256:
  `efc87556699abd6520921b0c888a395f2c95de2090e888ed5843ac35fc134e89`
- Selected Ks:
  `[7,18,3,7,1,9,3,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]`
- Decoder section bytes: `143,482`
- Brotli payload bytes: `143,422`
- Smoke rel_err vs quantized fp32: `0.056739`
- Max per-tensor smoke rel_err: `0.142895`
- Latent pairs decoded in smoke: `600`
- Runtime staging hygiene: smoke-created `__pycache__` directories were
  removed from `submission_dir/` after import verification; the final staged
  runtime tree contains only `inflate.py`, `inflate.sh`, `src/codec.py`, and
  `src/model.py`.

## Evidence class

`[CPU-build]`

No score, promotion, ranking, kill, or paper empirical claim is supported by
this artifact. The source planning manifest used a diagnostic/stub sensitivity
map; this makes the selected-K vector useful for wiring and bytes, not for
authoritative sensitivity claims.

Active dispatch blockers preserved in the build manifest:

- `apogee_int6_contest_cuda_anchor_required_first`
- `cpu_build_rel_err_proxy_not_score_evidence`
- `exact_cuda_auth_eval_not_yet_harvested`
- `requires_contest_auth_eval_json_before_score_promotion_rank_or_kill`
- `selected_Ks_json_cpu_planning_not_score_authority`
- `selected_Ks_source_blocker:diagnostic_or_stub_sensitivity_map_not_score_authority`
- `selected_Ks_source_blocker:requires_exact_cuda_auth_eval_before_score_claim`
- `selected_Ks_source_blocker:requires_static_archive_preflight`
- `selected_Ks_source_evidence_semantics:cpu_allocator_weight_export_no_score_no_dispatch`

Source-only blockers closed by this byte-closed rebuild:

- `selected_Ks_not_yet_encoded_in_no_dead_k_runtime_packet`
- `weight_export_only_no_byte_closed_archive`

## Composition review

- VStack status: `sensitivity planning -> selected-K allocation -> no-dead-K
  byte-closed pack` is now wired. The interface remains a typed manifest, not
  a hidden sidecar.
- HStack status: no new parallel payload stream exists. This candidate is a
  single monolithic archive transform, consistent with the PR101/PR106 archive
  anatomy finding.
- Synergy: the builder can consume beta-Fisher, Jacobian-pullback, or future
  certified boundary-mass selected-K manifests without source edits.
- Antagonism: diagnostic sensitivity can overfit the wrong tensor axis. The
  result is therefore blocked from dispatch until static preflight and a
  non-diagnostic sensitivity source, operator override, or explicit CUDA
  experiment rationale exists.
- Orthogonality: this changes decoded weights and therefore score risk. It is
  orthogonal to later entropy/packer work on the resulting monolithic stream.

Status: wiring landed; family open; no retirement or promotion.
