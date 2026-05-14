# PR106 Latent Sidecar + HLM1 Composition Review (2026-05-13)

## Summary

Reviewed the harvested Kaggle PR106 latent-score-table sidecar and tested the
obvious byte-closed composition: apply HLM1 fixed-latent recoding to that
sidecar archive. The composition is mechanically valid and full-frame
same-runtime parity passes, but it is not a score-lowering dispatch priority
against the current HLM1 exact frontier.

Current exact frontier remains:

- archive:
  `experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex/pr106_r2_hdm4_exact_cuda_hlm1_latent_candidate.zip`
- bytes: `186423`
- SHA-256:
  `8801845d5099b957898fb6c6e58625bfb4cc065085ed2e3154c2cbc702dc91e0`
- exact score: `0.20638030907530963 [contest-CUDA]`

## Harvested Sidecar Source

- source archive:
  `experiments/results/kaggle_pr106_latent_score_table_r2_20260511T151955Z/pr106_latent_score_table/latent_run/build/sidecar_archive.zip`
- bytes: `186822`
- SHA-256:
  `7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f`
- source inner PR106 SHA-256:
  `7f2cc905b7611ae8d7bced72be24e2266b0aa341f90cfeccbb0854fd8fc01eb7`
- sidecar bytes: `575`
- sidecar SHA-256:
  `acf58d03f708b0d297f9d32c2f47cefda3526f6665e19e9d0dee8073874c9d27`
- diagnostic score artifact:
  `experiments/results/kaggle_pr106_latent_score_table_r2_20260511T151955Z/pr106_latent_score_table/latent_run/eval/contest_auth_eval.json`
- diagnostic score: `0.2066238854574151 [diagnostic_cuda]`
- `score_claim=false`
- `promotion_eligible=false`

The Kaggle eval is useful signal, but it is not an exact leaderboard axis. It
must not unlock downstream sidechannel gates or submission decisions by itself.

## Audited Materialization

After hardening `tools/materialize_pr106_latent_score_table_candidate.py`, the
Kaggle table was re-materialized through the canonical proxy-evidence boundary:

```bash
.venv/bin/python tools/materialize_pr106_latent_score_table_candidate.py \
  --source-archive experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip \
  --score-table-npy experiments/results/kaggle_pr106_latent_score_table_r2_20260511T151955Z/pr106_latent_score_table/latent_run/score_table/score_table.npy \
  --score-table-manifest experiments/results/kaggle_pr106_latent_score_table_r2_20260511T151955Z/pr106_latent_score_table/latent_run/score_table/score_table_manifest.json \
  --output-dir experiments/results/pr106_latent_score_table_materialized_audited_20260513_codex \
  --delta-radius 2 \
  --top-k 600
```

Audit result:

- materialized archive:
  `experiments/results/pr106_latent_score_table_materialized_audited_20260513_codex/sidecar_archive.zip`
- bytes: `186822`
- SHA-256:
  `7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f`
- byte-identical to harvested Kaggle build: `true`
- manifest:
  `experiments/results/pr106_latent_score_table_materialized_audited_20260513_codex/materialization_manifest.json`
- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `target_modes=["contest_exact_eval_planning"]`
- dispatch blockers include:
  `kaggle_score_table_materialization_is_proxy_evidence_boundary`,
  `requires_lane_dispatch_claim_before_exact_eval`,
  `requires_exact_cuda_auth_eval_on_materialized_archive`,
  `requires_adjudicated_component_recompute_before_score_claim`

The score-table manifest itself had no missing custody fields. The only audit
warning is expected ZIP-container drift:
`source_archive_zip_sha256_differs_but_payload_sha256_matches`. The single
stored `0.bin` payload SHA matches the remote table provenance, so the table is
valid for materialization but still not score authority.

## Sidecar Recode Profile

Local profile artifact:
`experiments/results/pr106_latent_sidecar_recode_profile_r2_20260513_codex/profile.json`

Best lossless sidecar grammar:

- candidate: `pr101_ranked_no_op_sidecar_format_0x02`
- charged sidecar bytes: `533`
- delta vs current sidecar: `-42`
- rate-only score delta if runtime-consumed:
  `-0.000027966076031131196`
- semantic equivalence: `true`
- runtime decoder implemented: `true`

This primitive is real but too small to matter against the current HLM1
frontier. The previously exact-evaluated PR101-grammar low-level candidate is
also worse than HLM1:

- exact CUDA score:
  `0.2065174760196528 [contest-CUDA]`
- archive bytes: `186629`
- artifact:
  `experiments/results/modal_auth_eval/pr106_r2_pr101_grammar_lowlevel_repack_cuda_20260513_codex/contest_auth_eval.json`

## HLM1 Composition Candidate

Built locally with:

```bash
.venv/bin/python tools/build_pr106_hlm1_latent_candidate.py \
  --source-archive experiments/results/kaggle_pr106_latent_score_table_r2_20260511T151955Z/pr106_latent_score_table/latent_run/build/sidecar_archive.zip \
  --output-dir experiments/results/pr106_latent_sidecar_hlm1_candidate_20260513_codex \
  --source-label pr106_latent_sidecar_kaggle_r2 \
  --json-out experiments/results/pr106_latent_sidecar_hlm1_candidate_20260513_codex/manifest.json \
  --fail-if-blocked
```

Result:

- candidate archive:
  `experiments/results/pr106_latent_sidecar_hlm1_candidate_20260513_codex/pr106_latent_sidecar_kaggle_r2_hlm1_latent_candidate.zip`
- bytes: `186753`
- SHA-256:
  `67310908de56689a3883225efffa41d40461562c5d11017a5f3e626fac18c989`
- byte delta vs harvested sidecar source: `-69`
- rate-only score delta if components equal:
  `-0.00004594426776542982`
- ready for archive preflight: `true`
- exact eval dispatch: `false`

Proof artifacts:

- PacketIR identity:
  `experiments/results/pr106_latent_sidecar_hlm1_candidate_20260513_codex/packetir_identity.json`
- HLM1 runtime consumption:
  `experiments/results/pr106_latent_sidecar_hlm1_candidate_20260513_codex/hlm1_runtime_consumption.json`
- sidecar runtime consumption:
  `experiments/results/pr106_latent_sidecar_hlm1_candidate_20260513_codex/sidecar_runtime_consumption.json`
- same-runtime prefix parity:
  `experiments/results/pr106_latent_sidecar_hlm1_candidate_20260513_codex/same_runtime_prefix_parity_cpu.json`
- same-runtime full-frame parity:
  `experiments/results/pr106_latent_sidecar_hlm1_candidate_20260513_codex/same_runtime_full_parity_cpu.json`

Full-frame parity:

- axis label: `local-cpu-streaming-runtime`
- source frames: `1200`
- candidate frames: `1200`
- total bytes hashed per side: `3662409600`
- streaming SHA-256:
  `30bc014709737aa0a17aaef525183b13fefa6531ec3410ab84d91dc1199c387b`
- `full_frame_inflate_output_parity_claim=true`
- `score_claim=false`
- runtime SHA-256:
  `5fa58d34dbf195e960d9a3db6370bf238c1e4e459de6cf3f11487b0a1f4b272f`

## Classification

This is a measured composition negative against the current frontier, not a
method failure:

- HLM1 is a valid pure rate transform on the harvested sidecar archive.
- The sidecar bytes are runtime-consumed and frame-equivalent after HLM1.
- The composed archive is `330` bytes larger than the HLM1 non-promotional
  reference archive (`186753 - 186423`).
- Even granting the rate-only delta from the diagnostic sidecar artifact, the
  composition is not expected to beat `0.20638030907530963 [contest-CUDA]`.

Do not dispatch this exact-eval unless the goal is mechanism study rather than
score lowering.

## Next Score-Lowering Route

1. Keep HLM1 as a non-promotional `[contest-CUDA]` reference only; HDM4 remains
   the active exact dispatch frontier until explicit operator promotion changes
   that status.
2. Do not repeat generic sidecar recoding on this source; the best available
   consumed grammar is only `42` bytes.
3. If latent sidecars continue, move from fixed per-pair sidecar grammar to a
   learned or structured sidecar that changes distortion, not just bytes.
4. Prioritize PR95/HNeRV parity training and PacketIR transforms that change
   the high-byte structure or decoder payload, because HLM1/HDM4 evidence shows
   the remaining single-stream byte entropy is close to saturated.
