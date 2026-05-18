---
review_kind: implementation_readiness
review_id: z6_candidate4c_scorer_logit_trainer_readiness_20260518_codex
review_date: "2026-05-18"
lane_id: lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518
substrate: time_traveler_l5_z6_v2_candidate_4c_scorer_logit_conditioning
head_commit_at_review: c10dec618e49279ac709424c6aa180e8aede7293
evidence_axis: local_trainer_recipe_and_dry_run_readiness
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: false
---

# Z6 Candidate 4c scorer-logit conditioning readiness

## Classification

Candidate 4c moved from `PRISTINE probe / prose-only deferred ego source` to
`trainer + operator-recipe ready, non-spend`. No provider job was launched, no
lane claim was created, and no score/rank/promotion language is authorized by
this artifact.

This is the Atick side-info-channel branch from the Z6 Path B design memo
section 4.4c and the ATW V2 reactivation symposium Revision #5. It preserves
the operator fork: Candidate 1 re-fire versus Candidate 4c spend remains a
`NEEDS-OPERATOR-DECISION` dispatch choice.

## Code Surface

- `experiments/train_substrate_time_traveler_l5_z6.py`
  - `--ego-source` now accepts `scorer_logit`.
  - New `_derive_ego_motion_from_scorer_logits(...)` path uses compress-time
    SegNet logits plus PoseNet head features, reduces them to the existing
    fixed-width Z6 ego buffer, and records `ego_source` / `ego_motion_source`
    in archive metadata, provenance, and manifest.
  - Inflate remains scorer-free: only the reduced ego buffer is packed.
- `src/tac/tests/test_z6_v2_candidate_1_wave_2_build.py`
  - Parser coverage for `--ego-source scorer_logit`.
  - Fake PoseNet/SegNet unit test proving both scorer surfaces are consumed and
    reduced to `(N, ego_motion_dim)`.
  - Slot-budget regression coverage proving the default 8-dim Candidate 4c
    buffer preserves SegNet mean, PoseNet head, SegNet std, entropy, and
    margin instead of prefix-truncating the raw feature bank.
  - Current recipe state assertions reconciled with the operator-approved
    flip (`research_only=false`, `dispatch_enabled=true`, blockers cleared).
- `src/tac/tests/test_time_traveler_l5_z6_remote_driver.py`
  - Remote driver test now proves `Z6_EGO_SOURCE=scorer_logit` reaches trainer
    argv and provenance.

## Recipe Artifact

- `.omx/operator_authorize_recipes/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch.yaml`
  - `name`: `substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch`
  - `lane_id`: `lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518`
  - `research_only`: `false`
  - `dispatch_enabled`: `true`
  - `dispatch_blockers`: `[]`
  - `platform`: `modal`
  - `gpu`: `${MODAL_GPU:-T4}`
  - `cost_band.epochs`: `300`
  - `cost_band.predicted_cost_usd`: `13.0`
  - `predicted_band`: `[0.11, 0.17]`
  - `smoke_score_band`: `[0.13, 0.25]`
  - `remote_driver`: `scripts/remote_lane_substrate_time_traveler_l5_z6.sh`
  - Key env ladder:
    - `Z6_EGO_SOURCE=scorer_logit`
    - `Z6_PREDICTOR_ARCHITECTURE=single_layer_film_75k`
    - `Z6_PREDICTOR_PARAM_COUNT_TARGET=120000`
    - `Z6_EMIT_IDENTITY_PREDICTOR_DISAMBIGUATOR_ARCHIVE=true`
    - `Z6_TRAINER_MODE=full`

The recipe deliberately does not edit the Candidate 1 recipe. Candidate 1
remains the capacity-unwind branch; Candidate 4c is the side-info-channel
branch.

## Smoke Artifact

Local CPU smoke only:

```bash
.venv/bin/python experiments/train_substrate_time_traveler_l5_z6.py \
  --video-path upstream/videos/0.mkv \
  --output-dir experiments/results/z6_candidate4c_scorer_logit_smoke_codex_20260518T0145Z \
  --epochs 1 \
  --device cpu \
  --smoke \
  --ego-source scorer_logit \
  --predictor-architecture multi_layer_film_depth_3_300k \
  --emit-identity-predictor-disambiguator-archive
```

Observed smoke stats:

- `archive_bytes`: `52478`
- `identity_predictor_disambiguator_archive_bytes`: `52630`
- `ego_source`: `scorer_logit`
- `cfg.predictor_depth`: `3`
- `cfg.predictor_ego_motion_dim`: `4` (smoke tiny config)
- `evidence_grade`: `smoke-no-scorer`
- `score_claim_valid`: `false`
- `promotion_eligible`: `false`
- `ready_for_exact_eval_dispatch`: `false`

## Adversarial Review Finding

During post-launchability review I found a concrete Candidate 4c correctness
bug before provider spend:

- Finding: the first implementation concatenated `[seg_mean, pose, seg_std,
  entropy, margin]` and then truncated to `ego_motion_dim`.
- Impact: at the default `ego_motion_dim=8`, a realistic SegNet class width can
  consume the whole buffer with the `seg_mean` prefix, silently discarding
  PoseNet head, uncertainty, and margin signals. Even with smaller class counts,
  it can discard entropy/margin. This weakens the Atick side-info-channel
  hypothesis while still looking mechanically launchable.
- Fix: `experiments/train_substrate_time_traveler_l5_z6.py` now allocates a
  deterministic fixed-slot budget across every signal group. At the default
  8-dim buffer the slots are:
  - `seg_mean`: `2`
  - `pose`: `2`
  - `seg_std`: `2`
  - `entropy`: `1`
  - `margin`: `1`
- Mechanism: over-wide groups are reduced with deterministic adaptive average
  pooling instead of prefix truncation; under-wide groups are zero-padded.
- Custody hardening: the active slot budget is now written into all three
  artifact surfaces for `ego_source=scorer_logit`:
  - Z6PCWM1 archive `meta_blob`
  - `provenance.json`
  - `manifest.json`

This is a method-strengthening bugfix, not a score claim.

## Pair-Capped Real-Scorer Probe

After the slot-budget fix, a tiny full-path CPU probe exercised the corrected
`scorer_logit` derivation against the real scorer interfaces:

```bash
.venv/bin/python experiments/train_substrate_time_traveler_l5_z6.py \
  --video-path upstream/videos/0.mkv \
  --output-dir experiments/results/z6_candidate4c_slot_budget_paircapped_codex_20260518T0155Z \
  --epochs 1 \
  --batch-size 1 \
  --device cpu \
  --full-cpu \
  --advisory-cpu-explicitly-waived \
  --max-pairs 2 \
  --skip-auth-eval \
  --ego-source scorer_logit \
  --predictor-architecture single_layer_film_75k \
  --predictor-param-count-target 120000 \
  --predictor-ego-motion-dim 8 \
  --emit-identity-predictor-disambiguator-archive
```

Observed probe artifacts:

- `archive_bytes`: `189667`
- `archive_sha256`: `20382cad9701103a08e93b975b7e0db9ceaeab3ec26c6fc0c0273e9729ab65a5`
- `archive_zip_bytes`: `189316`
- `archive_zip_sha256`: `057b561724071b74fcc7b16c3100e3419b549beaa8130805fd2c822dca9fc82e`
- `ego_source`: `scorer_logit`
- `ego_motion_source`: `scorer_logit_segnet_logits_plus_posenet_head_standardized`
- `max_pairs`: `2`
- `research_only`: `true`
- `score_claim`: `false`
- `ready_for_exact_eval_dispatch`: `false`
- `auth_eval_skipped_reason`: `pair_capped_smoke_emits_truncated_raw_stream`

The probe is `[local CPU pair-capped smoke]` only. It proves scorer-interface
compatibility and archive emission after the slot-budget fix; it does not prove
score movement and must not be used for rank/promotion.

## Slot Metadata Custody Probe

After adding explicit slot metadata, I reran the tiny pair-capped full path:

```bash
.venv/bin/python experiments/train_substrate_time_traveler_l5_z6.py \
  --video-path upstream/videos/0.mkv \
  --output-dir experiments/results/z6_candidate4c_slot_metadata_paircapped_codex_20260518T0200Z \
  --epochs 1 \
  --batch-size 1 \
  --device cpu \
  --full-cpu \
  --advisory-cpu-explicitly-waived \
  --max-pairs 2 \
  --skip-auth-eval \
  --ego-source scorer_logit \
  --predictor-architecture single_layer_film_75k \
  --predictor-param-count-target 120000 \
  --predictor-ego-motion-dim 8 \
  --emit-identity-predictor-disambiguator-archive
```

Observed probe artifacts:

- `archive_bytes`: `189832`
- `archive_sha256`: `5a1cd10c71761c93f5e5fc6281e9a6ecec08e9fe4a0a84cc492ba9b3c72539ea`
- `archive_zip_bytes`: `189394`
- `archive_zip_sha256`: `9ac98a53a92f3fed6d1ba62f910bcdbee55c001811a243d85594ef4426e85e8c`
- `ego_source`: `scorer_logit`
- `ego_motion_source`: `scorer_logit_segnet_logits_plus_posenet_head_standardized`
- `scorer_logit_feature_slot_allocation_version`: `z6_candidate4c_v1`
- `scorer_logit_feature_slot_allocation`:
  - `seg_mean`: `2`
  - `pose`: `2`
  - `seg_std`: `2`
  - `entropy`: `1`
  - `margin`: `1`
- `research_only`: `true`
- `score_claim`: `false`
- `ready_for_exact_eval_dispatch`: `false`
- `auth_eval_skipped_reason`: `pair_capped_smoke_emits_truncated_raw_stream`

Verification command:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from tac.substrates.time_traveler_l5_z6.archive import parse_archive
arc = parse_archive(Path(
    "experiments/results/z6_candidate4c_slot_metadata_paircapped_codex_20260518T0200Z/0.bin"
).read_bytes())
print(arc.meta["scorer_logit_feature_slot_allocation"])
PY
```

Parsed archive metadata agreed with `provenance.json` and `manifest.json`.
This is still `[local CPU pair-capped smoke]` evidence only.

## Param-Envelope False-Positive Repair

The slot-budget custody probe also exposed an evidence bug: pair-capped local
probes were comparing the tiny `--max-pairs 2` parameter count against the full
Candidate 4c `120000` target. That produced a misleading warning:

- Actual pair-capped total: `103762`
- Reason: residual parameters shrink with `num_pairs=2` (`2 * latent_dim`)
- Dispatch-relevant full-equivalent total: `118114`
- Target: `120000`
- Full-equivalent deviation: `1.5717%`
- Verdict: within the ±5% envelope

Fix:

- `experiments/train_substrate_time_traveler_l5_z6.py` now records
  `param_count_target_diagnostic` with both actual and full-equivalent totals.
- Pair-capped probes compare the full-equivalent total against the recipe
  target, while preserving the actual tiny-run count.
- The diagnostic is written into:
  - Z6PCWM1 archive `meta_blob`
  - `provenance.json`
  - `manifest.json`

Verification probe:

```bash
.venv/bin/python experiments/train_substrate_time_traveler_l5_z6.py \
  --video-path upstream/videos/0.mkv \
  --output-dir experiments/results/z6_candidate4c_param_diag_paircapped_codex_20260518T0204Z \
  --epochs 1 \
  --batch-size 1 \
  --device cpu \
  --full-cpu \
  --advisory-cpu-explicitly-waived \
  --max-pairs 2 \
  --skip-auth-eval \
  --ego-source scorer_logit \
  --predictor-architecture single_layer_film_75k \
  --predictor-param-count-target 120000 \
  --predictor-ego-motion-dim 8 \
  --emit-identity-predictor-disambiguator-archive
```

Observed probe artifacts:

- `archive_bytes`: `190143`
- `archive_sha256`: `0be8da7d3629cfeff1de0b13bc95cc5c51e4bb7f03fe39c1d578c454a2a4d741`
- `archive_zip_bytes`: `189555`
- `archive_zip_sha256`: `05ae9870071cd81f3ba4d3cfaa367e4e71fb2898586cdfbe23db652d123049ff`
- `param_count_target_diagnostic.actual_total`: `103762`
- `param_count_target_diagnostic.full_equivalent_total`: `118114`
- `param_count_target_diagnostic.comparison_basis`: `full_equivalent_total_from_pair_capped_run`
- `param_count_target_diagnostic.within_5pct`: `true`
- `research_only`: `true`
- `score_claim`: `false`
- `ready_for_exact_eval_dispatch`: `false`

This is a false-authority repair. It does not change the Candidate 4c recipe
or make a score claim; it prevents a research-only probe from incorrectly
implying that the full dispatch underuses its capacity budget.

## Hidden-Width Remote-Driver Control Repair

Adversarial review found another pre-dispatch optimization gap: the trainer
already exposed `--predictor-hidden-dim` and
`--predictor-film-mlp-hidden-dim`, but the Z6 remote driver did not thread
either knob from recipe/env to trainer argv. That meant Candidate 4c could not
run single-layer capacity probes from an operator recipe without editing code.

Fix:

- `scripts/remote_lane_substrate_time_traveler_l5_z6.sh`
  - Adds `Z6_PREDICTOR_HIDDEN_DIM` default `64`.
  - Adds `Z6_PREDICTOR_FILM_MLP_HIDDEN_DIM` default `32`.
  - Writes both values to remote provenance.
  - Passes both flags to the trainer CLI.
- `experiments/train_substrate_time_traveler_l5_z6.py`
  - Adds both flags to `TIER_1_OPERATOR_REQUIRED_FLAGS` so operator recipes
    must preserve the env-to-CLI ladder.
- `.omx/operator_authorize_recipes/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch.yaml`
  - Pins current defaults explicitly:
    - `Z6_PREDICTOR_HIDDEN_DIM=64`
    - `Z6_PREDICTOR_FILM_MLP_HIDDEN_DIM=32`
  - Adds both CLI flags to `operator_required_flags`.
- `.omx/operator_authorize_recipes/substrate_z6_v2_candidate_1_multi_layer_film_modal_t4_smoke_dispatch.yaml`
  - Also pins the defaults and required flags so the broadened Z6 contract stays
    coherent. Candidate 1's depth-3 resolver still owns the final predictor
    hidden dim; this is contract hygiene, not a Candidate 1 architecture change.

This does not change Candidate 4c default behavior. It makes width tuning a
recipe/env decision for future sweeps instead of a code edit, which is necessary
for fast score-oriented search.

## Hidden-Width Pair-Capped Matrix

After the remote-driver control repair, I ran a local CPU pair-capped width
matrix with `--ego-source scorer_logit`, `--predictor-architecture
single_layer_film_75k`, `--predictor-ego-motion-dim 8`,
`--predictor-param-count-target 120000`, `--max-pairs 8`, `--epochs 1`, and
`--skip-auth-eval`.

Evidence grade for every row: `[local CPU pair-capped proxy]`,
`research_only=true`, `score_claim=false`, `promotion_eligible=false`,
`ready_for_exact_eval_dispatch=false`.

| width | result dir | val lag | archive bytes | archive sha256 | full-equivalent params | target deviation |
| ---: | --- | ---: | ---: | --- | ---: | ---: |
| 64 | `experiments/results/z6_candidate4c_width64_paircap8_codex_20260518T0214Z` | `96.2633209229` | `190260` | `094a0897b51b5d208164040a86640ec8503e31eb5e7b2dade4f100c4e0aaa144` | `118114` | `1.5716666667%` |
| 72 | `experiments/results/z6_candidate4c_width72_paircap8_codex_20260518T0214Z` | `96.2614440918` | `194764` | `122bb526097ec124790d207dcc82a2b118b446d0815b82fd9d7785a4fdf93830` | `120570` | `0.475%` |
| 80 | `experiments/results/z6_candidate4c_width80_paircap8_codex_20260518T0214Z` | `96.2644805908` | `199114` | `00fe8383a98bfbcb2fd1514b6cdd8e3677298583f29f0e1848d2976e8ce2a942` | `123026` | `2.5216666667%` |

Width 72 was the best pair-capped validation proxy by `0.0018768311` lag versus
width 64, but it costs `4504` additional archive bytes in this tiny run. This is
not enough evidence to auto-promote a recipe fork; it is a candidate for the
next Candidate 4c exact-smoke variant only if the operator chooses the 4c branch
over the still-open Candidate 1 refire. Width 80 is both worse and larger here.

## Archive-Meta Width Custody Repair

The first width-72 sidecar probe showed `provenance.json` and `manifest.json`
recording the requested/effective width fields, while the packed Z6PCWM1 archive
`meta_blob` still omitted them. That was a custody bug: the byte-closed artifact
must carry the same architecture-width contract as the sidecars.

Fix:

- `experiments/train_substrate_time_traveler_l5_z6.py` now writes
  `requested_predictor_hidden_dim`, `effective_predictor_hidden_dim`,
  `predictor_architecture`, and `predictor_depth` into the full archive
  `meta_blob`.
- The smoke archive metadata now mirrors those fields too, with
  `effective_predictor_hidden_dim` reflecting the deliberately tiny smoke config.

Post-fix verification probe:

```bash
.venv/bin/python experiments/train_substrate_time_traveler_l5_z6.py \
  --video-path upstream/videos/0.mkv \
  --output-dir experiments/results/z6_candidate4c_width72_archive_meta_paircap2_codex_20260518T0221Z \
  --epochs 1 \
  --batch-size 1 \
  --device cpu \
  --full-cpu \
  --advisory-cpu-explicitly-waived \
  --max-pairs 2 \
  --skip-auth-eval \
  --ego-source scorer_logit \
  --predictor-architecture single_layer_film_75k \
  --predictor-hidden-dim 72 \
  --predictor-film-mlp-hidden-dim 32 \
  --predictor-param-count-target 120000 \
  --predictor-ego-motion-dim 8 \
  --emit-identity-predictor-disambiguator-archive
```

Parsed `experiments/results/z6_candidate4c_width72_archive_meta_paircap2_codex_20260518T0221Z/0.bin`
with `tac.substrates.time_traveler_l5_z6.archive.parse_archive`:

- `archive_bytes`: `194747`
- `archive_sha256`: `cf914f273b562eff69d116366b2fc3b5e4acec4acfae2bc78c95762603465c16`
- `archive_zip_bytes`: `194119`
- `archive_zip_sha256`: `dc9bc244bd736698f30267ed590db57bb5de699c875524743f8d9f42bd58b059`
- `best_val_lag`: `96.8760070801`
- Archive `meta_blob`, `provenance.json`, and `manifest.json` agree on:
  - `requested_predictor_hidden_dim=72`
  - `effective_predictor_hidden_dim=72`
  - `predictor_film_mlp_hidden_dim=32`
  - `predictor_architecture=single_layer_film_75k`
  - `predictor_depth=1`
  - `param_count_target_diagnostic.full_equivalent_total=120570`
  - `param_count_target_diagnostic.within_5pct=true`

This remains `[local CPU pair-capped proxy]` evidence only. It is not a score
claim and is not rank/kill evidence for Candidate 4c.

## Predictor-Width Sidecar Alias Repair

Adversarial review found one remaining metadata asymmetry after the archive-meta
repair: `0.bin` carried the canonical `predictor_hidden_dim` alias, while
`provenance.json` and `manifest.json` carried only
`requested_predictor_hidden_dim` and `effective_predictor_hidden_dim`. A
downstream manifest consumer could therefore treat the sidecars as width-missing
even though the byte-closed artifact was correct.

Fix:

- `experiments/train_substrate_time_traveler_l5_z6.py` now centralizes
  predictor-width custody in `_predictor_width_metadata(...)`.
- The archive `meta_blob`, `provenance.json`, and `manifest.json` all consume
  the same helper output.
- The helper records:
  - `predictor_hidden_dim=<effective width>`
  - `requested_predictor_hidden_dim=<operator requested width>`
  - `effective_predictor_hidden_dim=<architecture-resolved width>`
  - `predictor_film_mlp_hidden_dim`
  - `predictor_architecture`
  - `predictor_depth`
- `src/tac/tests/test_z6_v2_candidate_1_wave_2_build.py` now has a regression
  test for the helper and asserts packed archive metadata includes the canonical
  `predictor_hidden_dim` alias.

Post-fix verification probe:

```bash
.venv/bin/python experiments/train_substrate_time_traveler_l5_z6.py \
  --video-path upstream/videos/0.mkv \
  --output-dir experiments/results/z6_candidate4c_width72_sidecar_alias_paircap2_codex_20260518T0227Z \
  --epochs 1 \
  --batch-size 1 \
  --device cpu \
  --full-cpu \
  --advisory-cpu-explicitly-waived \
  --max-pairs 2 \
  --skip-auth-eval \
  --ego-source scorer_logit \
  --predictor-architecture single_layer_film_75k \
  --predictor-hidden-dim 72 \
  --predictor-film-mlp-hidden-dim 32 \
  --predictor-param-count-target 120000 \
  --predictor-ego-motion-dim 8 \
  --emit-identity-predictor-disambiguator-archive
```

Parsed `experiments/results/z6_candidate4c_width72_sidecar_alias_paircap2_codex_20260518T0227Z/0.bin`
with `tac.substrates.time_traveler_l5_z6.archive.parse_archive`:

- `archive_bytes`: `194747`
- `archive_sha256`: `39b3d3f01c7d82d342f0a9fad826dceeae861717e84663175a068d82ebd84fd2`
- `archive_zip_bytes`: `194118`
- `archive_zip_sha256`: `aabaa955a9c2268883e0c7c75d5c286094148db074a4df7ffaecab9b2a854ae9`
- `best_val_lag`: `96.8760070801`
- `research_only=true`
- `score_claim=false`
- `ready_for_exact_eval_dispatch=false`
- Archive `meta_blob`, `provenance.json`, and `manifest.json` agree on:
  - `predictor_hidden_dim=72`
  - `requested_predictor_hidden_dim=72`
  - `effective_predictor_hidden_dim=72`
  - `predictor_film_mlp_hidden_dim=32`
  - `predictor_architecture=single_layer_film_75k`
  - `predictor_depth=1`
  - `param_count_target_diagnostic.full_equivalent_total=120570`
  - `param_count_target_diagnostic.within_5pct=true`

This is a custody/false-authority repair only. It preserves Candidate 4c launch
readiness but does not claim score movement.

## Full-Path Identity Disambiguator Emission Repair

Adversarial review found a dispatch-readiness bug: the
`--emit-identity-predictor-disambiguator-archive` flag was wired through the
operator recipe and remote driver, but the trainer honored it only in smoke
mode. A full Candidate 4c paid run would therefore have produced the primary
full-FiLM archive without the paired identity-control archive needed for the
probe-disambiguator arbitration.

Fix:

- `experiments/train_substrate_time_traveler_l5_z6.py` now emits
  `0_identity_predictor_disambiguator.bin` and
  `archive_identity_predictor_disambiguator.zip` in the full path when
  `--emit-identity-predictor-disambiguator-archive` is set.
- The full and identity-control archives share encoder, decoder, predictor blob,
  latent init, residuals, and ego-motion sections. The identity-control archive
  flips the inflate-time `identity_predictor` metadata and preserves the full
  predictor blob for rate-budget parity/custody.
- `provenance.json` and `manifest.json` now record identity-control archive
  bytes, SHA-256, ZIP bytes, and ZIP SHA-256.
- `src/tac/substrates/time_traveler_l5_z6/inflate.py` now permits identity-control
  archives to keep a full predictor state dict without loading it into the
  identity predictor module. This is intentional: the blob is parsed and paid
  for, while the runtime path is the identity ablation.
- Tests now cover both the full pair-capped trainer path and inflate-time
  identity-control consumption.

Post-fix verification probe:

```bash
.venv/bin/python experiments/train_substrate_time_traveler_l5_z6.py \
  --video-path upstream/videos/0.mkv \
  --output-dir experiments/results/z6_candidate4c_full_disambiguator_paircap2_codex_20260518T0234Z \
  --epochs 1 \
  --batch-size 1 \
  --device cpu \
  --full-cpu \
  --advisory-cpu-explicitly-waived \
  --max-pairs 2 \
  --skip-auth-eval \
  --ego-source scorer_logit \
  --predictor-architecture single_layer_film_75k \
  --predictor-hidden-dim 72 \
  --predictor-film-mlp-hidden-dim 32 \
  --predictor-param-count-target 120000 \
  --predictor-ego-motion-dim 8 \
  --emit-identity-predictor-disambiguator-archive
```

Parsed and inflated the identity-control archive:

- Primary full-FiLM archive:
  - path: `experiments/results/z6_candidate4c_full_disambiguator_paircap2_codex_20260518T0234Z/0.bin`
  - bytes: `194747`
  - SHA-256: `e83f8019690a9a87c1066a5791299dcc018bded9ab97f261db6444ba89b2f05b`
  - ZIP bytes: `194119`
  - ZIP SHA-256: `28b12f7cc136eb369e28df04135cca802fbbdec6eb73b07d22a796f570e11dce`
- Identity-control archive:
  - path: `experiments/results/z6_candidate4c_full_disambiguator_paircap2_codex_20260518T0234Z/0_identity_predictor_disambiguator.bin`
  - bytes: `195066`
  - SHA-256: `854a009ae8e9ab2e08c4faa1afcadc92615a40bad8d7586afc84366ba7ea1f11`
  - ZIP bytes: `194294`
  - ZIP SHA-256: `4298f449c7477704d979f2c6e46467665f0cb4ff4fc09bfa9a59de2a2b77228d`
- Parsed archive checks:
  - primary `identity_predictor=false`
  - identity-control `identity_predictor=true`
  - both archives carry `8` predictor state-dict keys
  - `identity_predictor_disambiguator=true` in identity-control metadata
- Inflate check:
  - command path: `tac.substrates.time_traveler_l5_z6.inflate.inflate_one_video(...)`
  - output: `experiments/results/z6_candidate4c_full_disambiguator_paircap2_codex_20260518T0234Z/identity_disambiguator_inflate_cpu.raw`
  - frames: `4`
  - raw bytes: `12208032`

Evidence grade: `[local CPU pair-capped proxy]`, `research_only=true`,
`score_claim=false`, `ready_for_exact_eval_dispatch=false`. The identity-control
archive is `319` bytes larger because it carries paired-control metadata; exact
disambiguator comparison must record both byte terms and component deltas rather
than treating the control as a score claim.

## Byte-Closed Disambiguator Probe Modernization

I upgraded `tools/probe_z6_predictive_coding_vs_identity_disambiguator.py` from
the old smoke-stats-only surface to a dual-mode probe:

- legacy mode: paired full/identity smoke `stats.json` comparison, still
  proxy-only;
- new archive mode: consumes full/identity Z6PCWM1 archive pairs, parses both
  archives, checks shared section parity, records ZIP sidecar custody, and stays
  fail-closed unless paired exact-eval JSONs are supplied.

Command run on the durable Candidate 4c pair-capped artifacts:

```bash
.venv/bin/python tools/probe_z6_predictive_coding_vs_identity_disambiguator.py \
  --run-dir experiments/results/z6_candidate4c_full_disambiguator_paircap2_codex_20260518T0234Z \
  --output-json .omx/research/z6_candidate4c_full_disambiguator_probe_20260518_codex.json \
  --output-md .omx/research/z6_candidate4c_full_disambiguator_probe_20260518_codex.md
```

Probe outputs:

- `.omx/research/z6_candidate4c_full_disambiguator_probe_20260518_codex.json`
- `.omx/research/z6_candidate4c_full_disambiguator_probe_20260518_codex.md`

Observed verdict:

- `verdict`: `pending_paired_exact_eval_json`
- `evidence_grade`: `byte_closed_archive_pair_no_score`
- `score_claim`: `false`
- `promotion_eligible`: `false`
- `ready_for_exact_eval_dispatch`: `false`
- blockers:
  - `no_paired_exact_eval_json`
  - `no_contest_cpu_cuda_pair`
  - `not_score_authority`

Archive-pair checks:

- `encoder_state_dict_equal=true`
- `decoder_state_dict_equal=true`
- `predictor_state_dict_equal=true`
- `predictor_keysets_equal=true`
- `latent_init_equal=true`
- `residuals_equal=true`
- `ego_motion_equal=true`

Byte deltas:

- identity minus full `0.bin` bytes: `319`
- identity minus full ZIP bytes: `175`
- identity minus full contest-rate term basis:
  `0.00011652531679637998`
- ZIP/member custody:
  - full ZIP members: `["0.bin"]`
  - identity ZIP members: `["0.bin"]`
  - full ZIP member SHA-256 matches sibling `0.bin`:
    `e83f8019690a9a87c1066a5791299dcc018bded9ab97f261db6444ba89b2f05b`
  - identity ZIP member SHA-256 matches sibling identity `0.bin`:
    `854a009ae8e9ab2e08c4faa1afcadc92615a40bad8d7586afc84366ba7ea1f11`

Adversarial repair after the first archive-mode landing:

- Finding: `--run-dir` mode parsed the sibling `0.bin` files while using
  `archive.zip` / `archive_identity_predictor_disambiguator.zip` as the contest
  byte/SHA basis. If a ZIP member drifted from its sibling `0.bin`, the probe
  could compare one byte stream and later match exact-eval custody against
  another.
- Fix: the probe now reads the ZIP sidecar member table, records member name,
  size, compressed size, CRC, SHA-256, and compression method, and blocks with
  `*_archive_zip_member_mismatch` if the sidecar member bytes do not match the
  sibling path parsed for pair checks.
- Additional exact-eval hardening: regression coverage now matches the observed
  `contest_auth_eval.json` schema where `archive_sha256` can live under
  `provenance.archive_sha256`, not only as a top-level field.
- Operator-surface fix: the Candidate 4c recipe now states the exact decision
  sign convention as `full_minus_identity_score <= -0.005` (equivalently
  `identity_minus_full_score >= 0.005`, since lower score wins), replacing the
  ambiguous `DeltaS >= 0.005` wording.

This is now a durable probe artifact, not a score artifact. Exact arbitration
still requires paired same-axis `contest_auth_eval.json` inputs whose archive
bytes/SHA match the ZIP sidecars.

## Paired Exact-Eval Production Path Repair

Follow-up review found one remaining production-path custody gap before any
Candidate 4c provider spend:

- Finding: the full trainer could emit the identity-control archive, but only
  the primary `archive.zip` had a canonical auth-eval JSON path. The full path
  also did not write `stats.json`, while the remote driver treats `stats.json`
  as the completion marker. A paid paired arbitration run could therefore
  produce the right bytes while losing the identity-control exact-eval result
  or being misclassified as incomplete.
- Fix: full-path Candidate 4c now resolves a second auth-eval path,
  `contest_auth_eval_identity_predictor_disambiguator.json`, runs the canonical
  auth-eval gate on `archive_identity_predictor_disambiguator.zip`, validates
  the identity result against the identity ZIP SHA-256, and records both primary
  and identity results into `provenance.json`, `manifest.json`, and `stats.json`.
- False-authority guard: when the identity-control archive is required,
  `auth_eval_score_claim_valid` is paired-gated. A valid primary exact-CUDA
  result alone can no longer mark the Candidate 4c run score-authoritative; the
  identity-control exact-CUDA JSON must also be present and valid.

Local non-authoritative verification:

```bash
.venv/bin/python experiments/train_substrate_time_traveler_l5_z6.py \
  --video-path upstream/videos/0.mkv \
  --output-dir experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z \
  --epochs 1 \
  --batch-size 1 \
  --device cpu \
  --full-cpu \
  --advisory-cpu-explicitly-waived \
  --max-pairs 2 \
  --skip-auth-eval \
  --ego-source scorer_logit \
  --predictor-architecture single_layer_film_75k \
  --predictor-hidden-dim 72 \
  --predictor-film-mlp-hidden-dim 32 \
  --predictor-param-count-target 120000 \
  --predictor-ego-motion-dim 8 \
  --emit-identity-predictor-disambiguator-archive
```

Observed local pair-capped artifact:

- output dir:
  `experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z`
- primary `0.bin` bytes: `194747`
- primary `0.bin` SHA-256:
  `4fd2e6cc2801bcb5ee9532b56219eb5468670b185b555d114b23ae30b59ae198`
- identity-control `0_identity_predictor_disambiguator.bin` bytes: `195066`
- identity-control ZIP SHA-256:
  `a2c47579f2d7e1e0c7b950e7ca7deca8fb0d8d414fd17a49d4b7946e19479d64`
- `stats_schema`: `time_traveler_l5_z6_full_stats_v1`
- `auth_eval_score_claim_valid`: `false`
- `primary_auth_eval_score_claim_valid`: `false`
- `identity_predictor_disambiguator_auth_eval_score_claim_valid`: `false`
- `paired_identity_auth_eval_required`: `true`
- `paired_identity_auth_eval_complete`: `false`
- `auth_eval_json_path`:
  `experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z/contest_auth_eval.json`
- `identity_predictor_disambiguator_auth_eval_json_path`:
  `experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z/contest_auth_eval_identity_predictor_disambiguator.json`

Evidence grade: `[local CPU pair-capped proxy; auth-eval skipped]`.
This is not a score claim, rank claim, or promotion artifact. It proves that
the full path now preserves the paired exact-eval custody surfaces needed for a
future claimed provider run.

## Run-Dir Exact-Eval Autodiscovery Repair

Adversarial handoff review found one more signal-loss seam between the repaired
trainer and the probe:

- Finding: the full trainer now writes canonical paired exact-eval JSON names,
  `contest_auth_eval.json` and
  `contest_auth_eval_identity_predictor_disambiguator.json`, but
  `tools/probe_z6_predictive_coding_vs_identity_disambiguator.py --run-dir`
  still required both JSON paths to be passed explicitly. A future provider
  harvest could therefore contain valid paired exact-eval JSONs while the probe
  stayed at `pending_paired_exact_eval_json` because the operator omitted two
  redundant flags.
- Fix: `--run-dir` now auto-discovers those default paired JSON names when both
  are present and feeds them into the exact-eval comparator.
- Fail-closed rule: if exactly one default JSON is present, the probe exits
  fatal with `run-dir paired exact-eval JSON missing ...` rather than treating a
  half-harvested run as pending/no-score. This prevents primary-only exact
  results from masquerading as a complete paired arbitration surface.

This is a harvest-path repair only. The probe remains non-authoritative for
promotion/rank/kill; it classifies paired exact-eval evidence and preserves
axis labels while leaving score-claim authority to the contest eval artifacts
and operator review.

## Remote Completion Missing-Stats Fail-Closed Repair

Remote-driver review found a completion-marker false-positive:

- Finding: `scripts/remote_lane_substrate_time_traveler_l5_z6.sh` treated a
  missing or malformed `stats.json` as
  `[training-artifact-no-score-claim] score_claim=false` and could still write
  `LANE_Z6_PCWM_DONE`. That is too soft for a provider run: if the trainer exits
  zero but the stats artifact is missing, the harvest is structurally broken,
  not merely non-authoritative.
- Fix: Stage 5 now requires `stats.json` to exist and parse as JSON before any
  completion marker is emitted. Missing stats exits `31`; malformed stats exits
  `32`. The existing marker logic is preserved only after stats custody is
  established.
- Impact: primary/identity paired exact-eval evidence cannot be silently lost
  behind a benign no-score completion banner. Autopilot/operator consumers now
  see a failed remote driver instead of a completed training artifact when the
  completion contract is broken.

This is a signal-preservation fix. It does not change model behavior, archive
bytes, or score authority.

## Remote Terminal-Claim Axis Split Repair

The missing-stats repair also exposed an operator-ledger ambiguity:

- Finding: a successful remote driver always appended terminal status
  `completed_z6_pcwm_remote_driver`, even when the completion marker was an
  explicit no-score training artifact. Terminal claim rows are consumed by
  dispatch monitors, so this status did not distinguish "remote driver produced
  a non-authoritative artifact" from "remote driver produced a contest-CUDA
  score-claim artifact."
- Fix: the terminal claim now splits success into:
  - `completed_z6_pcwm_remote_driver_contest_cuda_score_claim` only when the
    parsed stats marker is `[contest-CUDA]` and `score_claim=true`;
  - `completed_z6_pcwm_remote_driver_no_score_claim` for all other successful
    completions.
- Notes now include `evidence_marker=...`, `score_claim=...`, and the
  `stats_json=...` path, so terminal rows preserve the same axis/status used by
  `completion.log`.

This keeps the lane-claim ledger axis-labelled and prevents future harvest
automation from treating no-score completions as score-bearing terminal rows.

## Remote Completion Stale-Stats Reuse Repair

Adversarial review found a second completion-marker hazard:

- Finding: after the missing-stats check, a reused output directory could still
  contain a valid-looking `stats.json` from a previous invocation. If the
  current trainer exited zero without rewriting stats, Stage 5 would classify
  the stale file as the current run's evidence.
- Fix: before invoking the trainer, the remote driver moves any pre-existing
  `$Z6_OUTPUT_DIR/stats.json` to
  `$LOG_DIR/stale_stats_quarantine/stats.before_<job>.<utc>.<pid>.json`. Stage 4
  also records `REMOTE_DRIVER_STAGE4_STARTED_UNIX` immediately before invoking
  the trainer. Stage 5 rejects a trainer-written `stats.json` with mtime older
  than that timestamp and exits `33` before writing `LANE_Z6_PCWM_DONE`.
- Dispatch-ledger effect: stale stats terminalize as
  `failed_z6_pcwm_remote_driver_rc_33` with
  `evidence_marker=[not-yet-classified] score_claim=unknown`, preserving the
  fact that the current invocation did not produce usable evidence. A
  pre-existing quarantined stats file followed by no current stats terminalizes
  as `failed_z6_pcwm_remote_driver_rc_31`, with the old stats preserved for
  forensics.

This blocks a stale-output false authority path without changing model training
or archive bytes.

## Remote Pre-Existing Stats Quarantine Repair

The stronger no-stale-reuse invariant is now explicit:

- Any `stats.json` found before Stage 4 is previous-run evidence, not current
  evidence.
- The driver preserves it under `$LOG_DIR/stale_stats_quarantine/` instead of
  deleting it.
- The current invocation must produce a fresh `stats.json`; otherwise Stage 5
  fails closed as missing stats (`rc=31`).

This is important for reused Modal volumes and manual resume paths, where a
directory can outlive a single trainer invocation. It prevents stale evidence
reuse while preserving the old artifact for later forensic review.

## Catalog #324 Predicted-Band Provenance Repair

`tools/audit_predicted_band_provenance.py --strict` found one remaining
Catalog #324 violation in the Candidate 4c operator recipe:

- Recipe:
  `.omx/operator_authorize_recipes/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch.yaml`
- Failure:
  `predicted_band` was declared as `[0.11, 0.17]` without
  `predicted_band_validation_status`.
- Repair:
  add `predicted_band_validation_status: pending_post_training` and explicit
  reactivation criteria stating that the band is a planning prior until paired
  exact-eval evidence, full-vs-identity disambiguator evidence, and
  post-training Tier-C/byte-consumption validation land.

This preserves Candidate 4c launchability for an empirical smoke while blocking
false rank/kill, promotion, or frontier-language authority from an unvalidated
planning band.

## ASYMPTOTIC Readiness Queue Visibility Repair

The ASYMPTOTIC readiness queue had two stale operator-facing failures:

- Candidate 4c was launchable by recipe but absent from
  `tools/asymptotic_pursuit_candidate_readiness_assessment.py`.
- C6's original disabled random-init recipe could still surface stale
  predicted-band EV unless the Catalog #324 falsification text was interpreted
  manually.

Repair:

- Add `z6_v2_candidate_4c_scorer_logit` to the canonical candidate inventory.
- Map Candidate 4c to the existing Z6 trainer and its dedicated smoke-before-
  full recipe.
- Fall back to recipe-level `predicted_band` only as
  `PREDICTED_FROM_MODEL:<recipe>:predicted_band:<status>`.
- Suppress recipe bands when `predicted_band_validation_status` is
  `pending_post_training`, `dispatch_enabled=false`, and the reactivation
  criteria name the band as falsified.
- Look up lane maturity by exact recipe `lane_id` before substring fallback,
  read `impl_complete` from the canonical lane-maturity gate, and fail closed
  with `LANE_REGISTRY_NOT_REGISTERED` for future unregistered launch surfaces.

Lane registry repair:

- Registered `lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518`
  through `tools/lane_maturity.py`.
- Marked implementation, strict-test, memory, and runbook gates only.
- Left real-archive empirical, contest-CUDA, contest-CPU, and review gates
  unset. Candidate 4c is READY for an operator-authorized probe, not promotion.

Fresh readiness artifact:

- `.omx/state/asymptotic_pursuit/readiness_assessment_20260518T034050Z.json`
- TOP-1: `z6_v2_candidate_4c_scorer_logit`
- TOP-1 verdict: `READY`
- TOP-2: `time_traveler_l5_autonomy`
- Queue rollup: `ready_count=1`, `total_count=7`,
  `ready_total_cost_usd_if_dispatched=2.183`
- C6 original recipe: `DEFER`, predicted band suppressed by Catalog #324.

## Verification

- `bash -n scripts/remote_lane_substrate_time_traveler_l5_z6.sh`
  - PASS after remote completion missing-stats fail-closed repair
- `.venv/bin/python -m pytest src/tac/tests/test_time_traveler_l5_z6_remote_driver.py -q`
  - PASS after remote completion missing-stats fail-closed repair: `3 passed in 0.34s`
  - PASS after remote terminal-claim axis split repair: `3 passed in 0.32s`
  - PASS after remote completion stale-stats reuse repair:
    `4 passed in 0.41s`
  - PASS after remote pre-existing stats quarantine repair:
    `5 passed in 0.50s`
- `.venv/bin/python -m pytest src/tac/tests/test_z6_v2_candidate_1_wave_2_build.py src/tac/tests/test_time_traveler_l5_z6_remote_driver.py src/tac/tests/test_probe_z6_predictive_coding_vs_identity_disambiguator.py -q`
  - PASS after remote completion missing-stats fail-closed repair:
    `50 passed in 13.22s`
  - PASS after remote terminal-claim axis split repair:
    `50 passed in 11.88s`
  - PASS after remote completion stale-stats reuse repair:
    `51 passed in 12.55s`
  - PASS after remote pre-existing stats quarantine repair:
    `52 passed in 13.22s`
  - PASS after Catalog #324 predicted-band provenance repair:
    `52 passed in 12.05s`
- `.venv/bin/python tools/canonical_dispatch_optimization_protocol.py --trainer experiments/train_substrate_time_traveler_l5_z6.py --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch --json`
  - PASS after remote completion missing-stats fail-closed repair:
    `overall_pass=true`, `blockers=[]`, Tier 1/2/3 blockers all `[]`
  - PASS after remote terminal-claim axis split repair:
    `overall_pass=true`, `blockers=[]`, Tier 1/2/3 blockers all `[]`
  - PASS after remote completion stale-stats reuse repair:
    `overall_pass=true`, `blockers=[]`, Tier 1/2/3 blockers all `[]`
  - PASS after remote pre-existing stats quarantine repair:
    `overall_pass=true`, `blockers=[]`, Tier 1/2/3 blockers all `[]`
- `.venv/bin/python tools/audit_predicted_band_provenance.py --strict`
  - RED before Catalog #324 predicted-band provenance repair:
    `FAIL=1`, missing validation status on
    `substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch.yaml`
  - PASS after repair: `Recipes scanned: 81`, `In-scope: 20`, `PASS: 20`,
    `FAIL: 0`
- `.venv/bin/python -m pytest src/tac/tests/test_z6_v2_candidate_1_wave_2_build.py::test_candidate_4c_recipe_yaml_loads_and_is_distinct_launchable_lane src/tac/tests/test_z6_v2_candidate_1_wave_2_build.py::test_candidate_4c_recipe_passes_dispatch_optimization_protocol -q`
  - PASS after Catalog #324 predicted-band provenance repair:
    `2 passed in 0.57s`
- `.venv/bin/python -m pytest src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py -q --maxfail=1`
  - PASS after readiness queue visibility repair: `39 passed in 0.88s`
- `.venv/bin/python tools/asymptotic_pursuit_candidate_readiness_assessment.py --write-artifact`
  - PASS: wrote
    `.omx/state/asymptotic_pursuit/readiness_assessment_20260518T034050Z.json`;
    TOP-1 `z6_v2_candidate_4c_scorer_logit` READY; C6 DEFER with Catalog #324
    band suppression.
- `.venv/bin/python tools/asymptotic_pursuit_dispatch_queue.py --json`
  - PASS: `ready_count=1`, `total_count=7`,
    `ready_total_cost_usd_if_dispatched=2.183`.
- `.venv/bin/python tools/lane_maturity.py validate`
  - PASS after Candidate 4c lane registration: `OK - 860 lane(s) validated cleanly.`
- `git diff --check`
  - PASS after remote completion missing-stats fail-closed repair
- `.venv/bin/python -m py_compile tools/probe_z6_predictive_coding_vs_identity_disambiguator.py`
  - PASS after run-dir exact-eval autodiscovery repair
- `.venv/bin/python -m pytest src/tac/tests/test_probe_z6_predictive_coding_vs_identity_disambiguator.py -q`
  - PASS after run-dir exact-eval autodiscovery repair: `12 passed in 1.46s`
- `.venv/bin/python -m pytest src/tac/tests/test_z6_v2_candidate_1_wave_2_build.py src/tac/tests/test_time_traveler_l5_z6_remote_driver.py src/tac/tests/test_probe_z6_predictive_coding_vs_identity_disambiguator.py -q`
  - PASS after run-dir exact-eval autodiscovery repair: `49 passed in 12.00s`
- `.venv/bin/python tools/probe_z6_predictive_coding_vs_identity_disambiguator.py --run-dir experiments/results/z6_candidate4c_full_disambiguator_paircap2_codex_20260518T0234Z --output-json .omx/research/z6_candidate4c_full_disambiguator_probe_20260518_codex.json --output-md .omx/research/z6_candidate4c_full_disambiguator_probe_20260518_codex.md`
  - PASS after run-dir exact-eval autodiscovery repair:
    `verdict=pending_paired_exact_eval_json`,
    `evidence_grade=byte_closed_archive_pair_no_score`, because this local
    pair-capped artifact intentionally has no paired exact-eval JSONs.
- `.venv/bin/python tools/canonical_dispatch_optimization_protocol.py --trainer experiments/train_substrate_time_traveler_l5_z6.py --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch --json`
  - PASS after run-dir exact-eval autodiscovery repair: `overall_pass=true`,
    `blockers=[]`, Tier 1/2/3 blockers all `[]`
- `git diff --check`
  - PASS after run-dir exact-eval autodiscovery repair
- `tools/claim_lane_dispatch.py summary --live-only --format json`
  - `active_count=0`, `stale_nonterminal_count=0` at `2026-05-18T03:22:39Z`
  - `active_count=0`, `stale_nonterminal_count=0` at `2026-05-18T03:20:40Z`
  - `active_count=0`, `stale_nonterminal_count=0` at `2026-05-18T03:16:38Z`
  - `active_count=0`, `stale_nonterminal_count=0` at `2026-05-18T03:13:20Z`
  - `active_count=0`, `stale_nonterminal_count=0` at `2026-05-18T03:09:30Z`
  - `active_count=0`, `stale_nonterminal_count=0` at `2026-05-18T03:05:12Z`
- `.venv/bin/python -m py_compile experiments/train_substrate_time_traveler_l5_z6.py tools/probe_z6_predictive_coding_vs_identity_disambiguator.py`
  - PASS after paired exact-eval production-path repair
- `.venv/bin/python -m pytest src/tac/tests/test_z6_v2_candidate_1_wave_2_build.py src/tac/tests/test_time_traveler_l5_z6_remote_driver.py src/tac/tests/test_probe_z6_predictive_coding_vs_identity_disambiguator.py -q`
  - PASS after paired exact-eval production-path repair: `47 passed in 11.41s`
- `.venv/bin/python experiments/train_substrate_time_traveler_l5_z6.py ... --max-pairs 2 --skip-auth-eval --emit-identity-predictor-disambiguator-archive`
  - PASS after paired exact-eval production-path repair; wrote
    `experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z/stats.json`
- `.venv/bin/python -m py_compile experiments/train_substrate_time_traveler_l5_z6.py`
  - PASS
- `.venv/bin/python -m py_compile tools/probe_z6_predictive_coding_vs_identity_disambiguator.py`
  - PASS after byte-closed archive-pair modernization
- `.venv/bin/python -m pytest src/tac/tests/test_probe_z6_predictive_coding_vs_identity_disambiguator.py -q`
  - PASS after byte-closed archive-pair modernization: `9 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_probe_z6_predictive_coding_vs_identity_disambiguator.py src/tac/tests/test_z6_v2_candidate_1_wave_2_build.py -q`
  - PASS after ZIP-member custody and Candidate 4c sign-convention repair:
    `43 passed`
- `.venv/bin/python -m py_compile experiments/train_substrate_time_traveler_l5_z6.py src/tac/substrates/time_traveler_l5_z6/inflate.py`

## Diagnostic-only exact-eval handoff doctor repair

Follow-up Race Mode review found a stale readiness-model bug in
`tools/verify_candidate4c_launch_packet.py` after Candidate 4c was intentionally
split back to a diagnostic-only Modal training recipe:

- Live recipe state:
  `dispatch_enabled=false`, `smoke_only=true`,
  blocker `candidate4c_modal_training_recipe_is_diagnostic_only_exact_cuda_handoff_required`.
- Old doctor behavior: expected a queue-carried paid launch command and ran the
  Catalog #202 dirty-sentinel bypass probe even though paid training was no
  longer in scope. That produced misleading blockers such as stale sentinel
  audit hash and missing next paid command.
- Repair: the doctor now has two explicit modes:
  - `paid_training_launch_surface` for recipes that are actually dispatchable;
  - `diagnostic_only_exact_eval_handoff_required` for current Candidate 4c.
- In diagnostic-only mode the Catalog #202 bypass probe is skipped with a
  recorded reason, while the smoke-before-full dry-run, operator-authorize
  dry-run, active-claim cleanliness, local identity disambiguator proof, and
  exact-eval handoff status are still checked.
- The exact-eval handoff block now records four required future axes:
  full/identity `[contest-CUDA]` plus full/identity `[contest-CPU]`, each with a
  distinct lane id to avoid same-lane claim conflicts.
- The doctor emits only command templates until a full 600-pair archive/runtime
  packet exists. It no longer emits runnable Modal commands for the existing
  pair-capped local artifact.

Fresh no-spend artifact:

- `.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T103550Z.json`
- `current_mode`: `diagnostic_only_exact_eval_handoff_required`
- `checks_ok`: `true`
- `active_lane_claims_clean`: `true`
- `diagnostic_smoke_dry_run_ready`: `true`
- `provider_dispatch_attempted`: `false`
- `lane_claim_opened`: `false`
- `score_claim`: `false`
- `promotion_eligible`: `false`
- `exact_eval_handoff.ready_for_exact_eval_handoff`: `false`
- exact handoff blocker:
  `candidate4c_exact_handoff_latest_archive_pair_not_600_pairs`
- latest archive-pair evidence:
  `.omx/research/z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.json`
  with `latest_pair_count=2`, `evidence_grade=byte_closed_archive_pair_no_score`.

Interpretation: Candidate 4c is not dead and not dispatch-authoritative. The
pair-capped archive pair proves byte-closed full-vs-identity consumption and
runtime-output difference, but the score-moving next artifact is still a
harvested full 600-pair archive/runtime packet, followed by paired exact eval
on both axes. No provider spend, no lane claim, and no score/rank/promotion
claim occurred in this repair.

Verification after this repair:

- `.venv/bin/python -m py_compile tools/verify_candidate4c_launch_packet.py src/tac/tests/test_verify_candidate4c_launch_packet.py`
  - PASS
- `.venv/bin/python -m pytest -q src/tac/tests/test_verify_candidate4c_launch_packet.py -x`
  - PASS: `10 passed in 0.17s`
- `.venv/bin/python tools/verify_candidate4c_launch_packet.py --json --write-artifact`
  - Expected rc=1 because no full 600-pair archive pair exists yet.
  - Artifact blocker set reduced to the real handoff blockers listed above.
  - PASS after full-path identity-disambiguator repair
- `.venv/bin/python -m pytest src/tac/tests/test_z6_v2_candidate_1_wave_2_build.py -q`
  - PASS: `23 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_time_traveler_l5_z6_remote_driver.py src/tac/substrates/time_traveler_l5_z6/tests/test_multi_layer_film_predictor.py -q`
  - PASS: `15 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_time_traveler_l5_z6_remote_driver.py src/tac/tests/test_z6_v2_candidate_1_wave_2_build.py -q`
  - PASS before slot-budget fix: `28 passed`
  - PASS after slot-budget fix: `30 passed`
  - PASS after slot-metadata custody hardening: `31 passed`
  - PASS after param-envelope false-positive repair: `32 passed`
  - PASS after hidden-width remote-driver control repair: `32 passed`
  - PASS after archive-meta width custody repair: `33 passed`
  - PASS after predictor-width sidecar alias repair: `34 passed`
- `.venv/bin/python -m pytest src/tac/substrates/time_traveler_l5_z6/tests/test_z6.py src/tac/tests/test_time_traveler_l5_z6_remote_driver.py src/tac/tests/test_z6_v2_candidate_1_wave_2_build.py -q`
  - PASS after full-path identity-disambiguator repair: `71 passed`
- `.venv/bin/python tools/canonical_dispatch_optimization_protocol.py --trainer experiments/train_substrate_time_traveler_l5_z6.py --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch --json`
  - PASS before and after slot-budget fix: `overall_pass=true`,
    `blockers=[]`, Tier 1/2/3 blockers all `[]`
  - PASS after slot-metadata custody hardening: `overall_pass=true`,
    `blockers=[]`, Tier 1/2/3 blockers all `[]`
  - PASS after hidden-width remote-driver control repair: `overall_pass=true`,
    `blockers=[]`, Tier 1/2/3 blockers all `[]`
  - PASS after archive-meta width custody repair: `overall_pass=true`,
    `blockers=[]`, Tier 1/2/3 blockers all `[]`
  - PASS after predictor-width sidecar alias repair: `overall_pass=true`,
    `blockers=[]`, Tier 1/2/3 blockers all `[]`
  - PASS after full-path identity-disambiguator repair: `overall_pass=true`,
    `blockers=[]`, Tier 1/2/3 blockers all `[]`
- `.venv/bin/python tools/canonical_dispatch_optimization_protocol.py --trainer experiments/train_substrate_time_traveler_l5_z6.py --recipe substrate_z6_v2_candidate_1_multi_layer_film_modal_t4_smoke_dispatch --json`
  - PASS after hidden-width remote-driver control repair: `overall_pass=true`,
    `blockers=[]`, Tier 1/2/3 blockers all `[]`
  - PASS after archive-meta width custody repair: `overall_pass=true`,
    `blockers=[]`, Tier 1/2/3 blockers all `[]`
  - PASS after predictor-width sidecar alias repair: `overall_pass=true`,
    `blockers=[]`, Tier 1/2/3 blockers all `[]`
  - PASS after full-path identity-disambiguator repair: `overall_pass=true`,
    `blockers=[]`, Tier 1/2/3 blockers all `[]`
- `.venv/bin/python tools/operator_authorize.py --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch --dry-run --yes`
  - PASS: printed operator banner, cost band `$5.00/$10.00/$20.00`
    hand-calibrated fallback, then `--dry-run; no confirmation prompt, no dispatch`
  - PASS after full-path identity-disambiguator repair: same no-spend dry-run
    behavior
- `.venv/bin/python tools/run_modal_smoke_before_full.py --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch --operator-handle codex:z6_candidate4c_scorer_logit --dry-run`
  - PASS: `--dry-run; no Modal dispatch`; would dispatch 100-epoch T4 smoke and
    full only after smoke green
- `git diff --check`
  - PASS after archive-meta width custody repair
  - PASS after predictor-width sidecar alias repair
  - PASS after full-path identity-disambiguator repair
- `tools/claim_lane_dispatch.py summary --live-only --format json`
  - `active_count=0`, `stale_nonterminal_count=0` at `2026-05-18T02:02:01Z`
  - `active_count=0`, `stale_nonterminal_count=0` at `2026-05-18T02:22:08Z`
  - `active_count=0`, `stale_nonterminal_count=0` at `2026-05-18T02:28:00Z`
  - `active_count=0`, `stale_nonterminal_count=0` at `2026-05-18T02:35:11Z`

## Remaining Blockers

1. Candidate 4c now has a dedicated operator-authorize recipe and smoke-before-
   full dry-run path. Actual provider spend still requires explicit operator
   dispatch choice, a fresh lane claim, and the recipe's claim lifecycle.
2. The smoke is not a scorer-logit empirical result. Smoke mode remains
   scorer-free by design and records `smoke_proxy_no_scorer_load`.
3. Candidate 1 versus Candidate 4c is still an operator fork. The existing
   Candidate 1 run was refused pre-provider by Catalog #313 predecessor state;
   I did not override that state because it would silently pick Candidate 1.
4. Candidate 4c exact evidence still requires a fresh dispatch claim before any
   provider job, then paired smoke/full harvest with full-FiLM and identity-
   control archive SHA, bytes, runtime tree SHA, component distances, axis
   labels, and terminal claim rows.

## Six-Hook Wire-In

1. Sensitivity-map contribution: `scorer_logit` captures SegNet logit mean/std,
   entropy, margin, and PoseNet head features as a richer side-info channel.
2. Pareto constraint: no spend from this artifact; next Pareto decision remains
   Candidate 1 cheap re-fire versus Candidate 4c higher-leverage scorer-logit.
3. Bit-allocator hook: no archive grammar expansion; existing ego buffer is the
   byte surface, so rate impact stays bounded by current Z6PCWM1 side-info bytes.
4. Cathedral/autopilot dispatch hook: remote driver now proves
   `Z6_EGO_SOURCE=scorer_logit` threads into trainer argv and provenance.
5. Continual-learning posterior update: none; no score anchor produced.
6. Probe-disambiguator: active path is Candidate 4c full-FiLM versus identity
   predictor after provider execution; local pair-capped evidence now proves the
   full trainer emits and inflates both archive surfaces, but exact arbitration
   still needs same-axis component deltas.

## 2026-05-18 Codex Distinguishing-Byte Consumption Proof

Candidate 4c's recipe no longer leaves `byte_mutation_smoke_passes` as pending.
The current byte-closed local packet
`experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z/archive.zip`
was parsed as Z6PCWM1 and the scorer-logit ego-motion section was identified
as `0.bin@192390:16` (`ego_motion_len=16`, `num_pairs=2`, `ego_dim=8`).

Proof command:

```bash
.venv/bin/python tools/verify_distinguishing_feature_byte_mutation.py \
  --archive experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z/archive.zip \
  --inflate-sh experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z/submission_dir/inflate.sh \
  --distinguishing-byte-range scorer_logit_ego_motion=0.bin@192390:16 \
  --archive-staging-mode extracted_members \
  --mutations-per-section 4 \
  --output-json experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z/scorer_logit_ego_motion_byte_mutation_proof.json \
  --verbose
```

Result: `PASSED`. The repeated baseline inflate was deterministic with
aggregate SHA-256
`bfe7db19adc1623760891628c652dfc1f70dbf7b279a2658028b4c1248172c24`; 2 of 4
single-byte mutations in the scorer-logit ego section changed the inflated
output, first changed aggregate SHA-256
`0fbd70338f8b522d0c73a4f054e942687ce916ab5524a449ca09d0cbaeeae7bf`.

Harness bug found and fixed: the generic distinguishing-feature verifier staged
`archive.zip` into `archive_dir`, which is wrong for contest-style runtimes that
expect extracted members such as `archive_dir/0.bin`. The verifier now has an
explicit `--archive-staging-mode extracted_members` path plus a regression test
for that contract. A second verifier reproducibility bug was also fixed: the
proof no longer depends on a caller-provided `PYTHON` shell variable. The
verifier exports `PYTHON` to the interpreter running the verifier by default,
and the proof JSON records
`inflate_python=/Users/adpena/Projects/pact/.venv/bin/python`. This proof is
byte-consumption/no-op evidence only:
`score_claim=false`, `promotion_eligible=false`, and provider dispatch still
requires explicit operator choice, a fresh lane claim, and paired CPU/CUDA
harvest.

## 2026-05-18 Codex Queue Recheck After TT5L Blocker Reconciliation

`tools/asymptotic_pursuit_dispatch_queue.py --json` still ranks
`z6_v2_candidate_4c_scorer_logit` as the only current `READY`
asymptotic-pursuit candidate after the stale TT5L probe-verdict hash blocker
was removed from the TT5L recipe. The Candidate 4c smoke-before-full actuator
still resolves without provider spend:

```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
  --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch \
  --operator-handle codex:z6_candidate4c_scorer_logit \
  --dry-run
```

Dry-run plan: recipe resolves to
`.omx/operator_authorize_recipes/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch.yaml`,
`epoch_env_var=Z6_EPOCHS`, `smoke_validation_contract=contest_cuda_auth_eval_v1`,
100-epoch T4 smoke, 1.0h timeout, full dispatch only after smoke green. No
Modal work was spawned and no lane claim was opened. This is launch-surface
evidence, not score evidence.

## 2026-05-18 Codex Contest-Target Metadata Guard

Adversarial review found one launch-surface mismatch: Candidate 4c was marked
`dispatch_enabled=true` and ranked `READY`, but its `target_modes` list did not
carry the explicit `contest_exact_eval` marker required by the provider-runtime
contract for paid contest-targeted queues. The recipe now includes
`contest_exact_eval` alongside `contest_one_video_replay`, `contest_generalized`,
and `research_substrate`.

`tools/asymptotic_pursuit_candidate_readiness_assessment.py` now fails closed
with `RECIPE_target_modes_missing_contest_exact_eval` for any dispatch-enabled,
non-research recipe that lacks this target marker. This keeps the queue from
turning a research-target-only recipe into a paid exact-eval recommendation.
Candidate 4c remains `READY` only because the recipe now explicitly targets the
contest exact-eval path. This is metadata/custody hardening only; no score claim
or provider dispatch was made.

## 2026-05-18 Codex Horizon-Class Consistency Guard

Adversarial queue review found a second launch-surface mismatch: Candidate 4c's
recipe still advertised `horizon_class: frontier_pursuit` after the live
readiness classifier moved it into `asymptotic_pursuit` from the recipe-level
`predicted_band: [0.11, 0.17]`. The recipe now carries
`horizon_class: asymptotic_pursuit`, matching the classifier and the dispatch
queue entry.

`tools/asymptotic_pursuit_candidate_readiness_assessment.py` now fails closed
with `RECIPE_horizon_class_mismatch:<recipe>!=<computed>` for dispatch-enabled,
non-research recipes whose declared horizon class disagrees with the computed
class. This prevents a stale operator recipe label from staying launchable after
the predicted band changes.

Verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py -q
# 43 passed in 0.93s

.venv/bin/python tools/asymptotic_pursuit_candidate_readiness_assessment.py --json \
  | jq '.candidates[] | select(.substrate_id=="z6_v2_candidate_4c_scorer_logit") | {horizon_class, readiness_verdict, blocking_issues}'
# {"horizon_class":"asymptotic_pursuit","readiness_verdict":"READY","blocking_issues":[]}

.venv/bin/python tools/asymptotic_pursuit_dispatch_queue.py --json \
  | jq '{top_1_substrate, top_1_readiness_verdict, candidate4c:([.dispatch_sequence[] | select(.substrate_id=="z6_v2_candidate_4c_scorer_logit") | {horizon_class, readiness_verdict, blocking_issues}] | .[0])}'
# top_1_substrate=z6_v2_candidate_4c_scorer_logit, candidate4c.horizon_class=asymptotic_pursuit, candidate4c.blocking_issues=[]
```

Additional custody checks:

```bash
.venv/bin/python tools/lane_maturity.py validate
# OK — 867 lane(s) validated cleanly.

.venv/bin/python tools/audit_predicted_band_provenance.py --strict
# PASS: 20, FAIL: 0

git diff --check
# no output

.venv/bin/python tools/claim_lane_dispatch.py summary
# CLAIM_SUMMARY active=0 stale_nonterminal=0 terminal_latest=942 unparsable_timestamp=0 invalid_lane_id=4

.venv/bin/python tools/run_modal_smoke_before_full.py \
  --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch \
  --operator-handle codex:z6_candidate4c_scorer_logit \
  --dry-run
# --dry-run; no Modal dispatch; would dispatch 100-epoch T4 smoke only after explicit operator path.
```

No provider job was launched and no lane claim was opened.

## 2026-05-18 exact-eval handoff command hygiene follow-up

After the full-600 zero-epoch archive pair was produced, the Candidate 4c
packet doctor was hardened so post-harvest eval cannot drift back into
single-axis Modal wrapper commands. The handoff surface now points at
`tools/dispatch_modal_paired_auth_eval.py` for both full and identity archives,
preserving paired `[contest-CUDA]` plus `[contest-CPU]` execution under one
command per archive mode.

Current artifact:

- `.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T105922Z.json`
- `exact_eval_handoff.ready_for_exact_eval_handoff=true`
- `exact_eval_handoff.latest_pair_count=600`
- `ready_for_operator_paid_execution=false`
- `provider_dispatch_attempted=false`
- `lane_claim_opened=false`
- `score_claim=false`
- `promotion_eligible=false`

Interpretation: Candidate 4c has a byte-closed, post-harvest exact-eval
handoff packet, but the diagnostic Modal training recipe remains disabled for
paid training and should not be used as a contest-CUDA score claim surface.

## 2026-05-18 Codex full-600 zero-epoch exact-eval handoff repair

The earlier no-spend launch packet correctly blocked Candidate 4c exact-eval
handoff because the latest full/identity archive pair carried only `2` parsed
pairs. I ran a local CPU, zero-epoch, no-auth-eval full-path materialization to
separate "missing packet" from "full-path runtime failure".

New artifacts:

- local packet:
  `experiments/results/z6_candidate4c_full600_zeroepoch_packet_codex_20260518T104201Z/`
- refreshed disambiguator:
  `.omx/research/z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.json`
- refreshed queue:
  `.omx/state/asymptotic_pursuit/dispatch_queue_20260518T104919Z.json`
- refreshed no-spend packet:
  `.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T104931Z.json`
- dedicated ledger:
  `.omx/research/z6_candidate4c_full600_zeroepoch_handoff_20260518_codex.md`

Full archive custody:

- parsed pairs: `600`
- ZIP bytes: `211866`
- ZIP sha256:
  `5b371490b4459b85e95e6173653fc1b9aa78010681862ec51111166a6c867c4b`

Identity archive custody:

- parsed pairs: `600`
- ZIP bytes: `212047`
- ZIP sha256:
  `e6cd9bf67ca68bcdf93aa0e804435b75b813e420d5e3964b3a6cb6cee28e3589`

Local full-vs-identity inflate-output proof:

- `runtime_output_changed=true`
- `total_byte_differences=33048720`
- full output aggregate sha256:
  `241f9cf0d6234a728a165173e0f352beb5254d358dacf0e6d7ff027b0f58c712`
- identity output aggregate sha256:
  `5c0673169daabf7a90cddaa86b23b157019f96c63f68daa36eed786be368d94e`

The handoff status now reports:

- `exact_eval_handoff.ready_for_exact_eval_handoff=true`
- `exact_eval_handoff.latest_pair_count=600`
- `ready_for_operator_paid_execution=false`
- blockers:
  `candidate4c_recipe_dispatch_disabled_exact_eval_handoff_required`,
  `candidate4c_paid_training_launch_not_in_scope_recipe_dispatch_disabled`

Interpretation: the full-600 archive/runtime packet exists and is byte-closed
for exact-eval handoff planning. It is not a score claim because the run used
`--epochs 0`, skipped auth eval, and has no paired contest CPU/CUDA JSON.
Provider spend still requires fresh lane claims on the generated exact-eval
commands, not launching the diagnostic-only Modal training recipe as if it were
contest-CUDA authority.

## 2026-05-18 Codex Candidate 4c Modal Diagnostic-Only Split

The Catalog #271 blocker was resolved by splitting authority domains:

- Modal training recipe:
  `.omx/operator_authorize_recipes/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch.yaml`
- current queue:
  `.omx/state/asymptotic_pursuit/dispatch_queue_20260518T095039Z.json`
- current no-spend packet:
  `.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T095046Z.json`
- fresh Codex review:
  `.omx/state/candidate4c_launch_packet/candidate4c_codex_pre_dispatch_review_20260518T0952Z.json`

Current status:

- `ready_for_paid_dispatch_count=0`
- Candidate 4c remains TOP-1 but `top_1_readiness_verdict=NEEDS_FIX`
- Candidate 4c blockers:
  `RECIPE_dispatch_enabled=false`,
  `RECIPE_DISPATCH_BLOCKER:candidate4c_modal_training_recipe_is_diagnostic_only_exact_cuda_handoff_required`
- recipe is `smoke_only: true`, `smoke_validation_contract: training_artifact_v1`
- recipe no longer targets `contest_exact_eval`
- recipe uses `Z6_MAX_PAIRS=64` and `Z6_SKIP_AUTH_EVAL=1`
- Modal metadata is now 100 epochs, matching the smoke wrapper
- latest Codex pre-dispatch review verdict is `approve`
- latest no-spend packet remains `ready_for_operator_paid_execution=false`
- provider dispatch attempted: false
- lane claim opened: false
- score claim: false
- promotion eligible: false

Implementation notes:

- `scripts/remote_lane_substrate_time_traveler_l5_z6.sh` now threads
  `Z6_MAX_PAIRS` and `Z6_SKIP_AUTH_EVAL`; pair-capped runs automatically add
  `--skip-auth-eval`.
- `experiments/train_substrate_time_traveler_l5_z6.py` now writes the
  recipe/runtime lane id from `Z6_LANE_ID` into archive/provenance manifests.
- `tools/run_modal_smoke_before_full.py` accepts `pair_capped_smoke` as a
  valid `training_artifact_v1` false-authority smoke mode.
- `tools/verify_candidate4c_launch_packet.py` now checks the diagnostic
  smoke budget shape (`$1.250`) instead of the retired `$13` full/eval path.

Exact-CUDA authority is not restored by this patch. The next promotable path
requires a separate 600-pair archive/runtime packet plus a claimed canonical
exact-eval handoff, e.g. `experiments/modal_auth_eval.py` with provider-level
`modal run --detach` and wrapper `--detach --provider-detach-ack`.

Verification:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_run_modal_smoke_before_full.py \
  src/tac/tests/test_z6_v2_candidate_1_wave_2_build.py \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py \
  src/tac/tests/test_verify_candidate4c_launch_packet.py \
  src/tac/tests/test_check_271_pre_dispatch_codex_review.py
# 197 passed in 29.68s

.venv/bin/python tools/run_codex_review_for_dispatch.py \
  --trainer experiments/train_substrate_time_traveler_l5_z6.py \
  --recipe .omx/operator_authorize_recipes/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch.yaml \
  --estimated-cost-usd 1.25 \
  --skip-cache \
  --json-out .omx/state/candidate4c_launch_packet/candidate4c_codex_pre_dispatch_review_20260518T0952Z.json \
  --timeout-seconds 600
# verdict=approve; findings=0

.venv/bin/python tools/asymptotic_pursuit_dispatch_queue.py --json --write-artifact
# wrote dispatch_queue_20260518T095039Z.json; ready_for_paid_dispatch_count=0

.venv/bin/python tools/verify_candidate4c_launch_packet.py --write-artifact
# rc=1 by design; wrote candidate4c_no_spend_launch_packet_20260518T095046Z.json
# ready_for_operator_paid_execution=false
```

No provider job was launched and no lane claim was opened.

## 2026-05-18 Codex Catalog #202 Dirty-Sentinel Queue Repair

Adversarial review found a second paid-launch false-runnable gap in
`tools/asymptotic_pursuit_dispatch_queue.py`: the queue previously treated the
Catalog #202 paired env vars as sufficient when the shared worktree was dirty
unless the newest persisted sentinel audit said an effective Modal sentinel was
dirty. That is weaker than `tools/operator_authorize.py`'s actual runtime
contract. If an operator edited a sentinel after the last audit, or if no audit
artifact existed for the current dirty sentinel bytes, the queue could mark a
paid row as `immediately_runnable_paid_launch=true` even though
`operator_authorize.py` would exit 12 before provider setup.

Repair:

- `tools/asymptotic_pursuit_dispatch_queue.py` now computes the current
  effective Modal sentinel snapshot from the recipe, current git status, and
  current file SHA-256s.
- Dirty sentinel files require
  `OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON` plus a latest audit whose
  `sentinel_set_sha256` and dirty sentinel path list match the current
  snapshot.
- Audit-backed paid-launch commands are emitted only when that latest audit is
  current. Otherwise the row remains READY but not immediately runnable, with
  `CATALOG_202_dirty_sentinel_requires_current_audit_json_before_paid_dispatch`
  in `paid_launch_missing_preconditions`.

Live Candidate 4c snapshot after the repair:

```bash
.venv/bin/python tools/audit_catalog202_sentinel_cleanliness.py \
  --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch \
  --json
# dirty_sentinel_paths=[
#   "tools/operator_authorize.py",
#   "tools/probe_z6_predictive_coding_vs_identity_disambiguator.py",
#   "tools/run_modal_smoke_before_full.py"
# ]
# sentinel_set_sha256=a04c09b40cca80dd98c41968770ce7d2b19b672e2553d6a2bbb5c03b1a5aa387
# ready_for_catalog202_audit_backed_dirty_sentinel_attestation=true
```

Verification:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py \
  src/tac/tests/test_check_202_operator_authorize_clean_bypass.py \
  src/tac/tests/test_run_modal_smoke_before_full.py -q
# 111 passed in 15.97s

.venv/bin/python tools/asymptotic_pursuit_dispatch_queue.py --json
# ready_for_paid_dispatch_count=1
# immediately_runnable_paid_dispatch_count=0
# Candidate 4c current_sentinel_snapshot_valid=true
# Candidate 4c dirty_sentinel_audit_required=true
# latest_sentinel_audit_matches_current=true
# env_sentinel_audit_matches_current=false
# paid_launch_missing_preconditions=[
#   CATALOG_202_dirty_worktree_requires_paired_env_attestation_before_paid_dispatch,
#   CATALOG_202_dirty_sentinel_requires_current_audit_json_before_paid_dispatch
# ]
# audit_backed_paid_launch_command present, but paid launch still requires
# OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK plus the paired operator
# attestation env before it becomes immediately runnable.

OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1 \
  OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=operator-attests \
  OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON=.omx/state/catalog202_sentinel_cleanliness/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch_20260518T065922Z.json \
  .venv/bin/python tools/asymptotic_pursuit_dispatch_queue.py --json
# stale env audit path remains blocked even though latest_sentinel_audit_matches_current=true:
# env_sentinel_audit_matches_current=false
# immediately_runnable_paid_launch=false

OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1 \
  OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=operator-attests \
  OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON=.omx/state/catalog202_sentinel_cleanliness/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch_20260518T072226Z.json \
  .venv/bin/python tools/asymptotic_pursuit_dispatch_queue.py --json
# current env audit path is accepted:
# env_sentinel_audit_matches_current=true
# immediately_runnable_paid_launch=true
```

Follow-up env-path refinement:

- The queue now separates `latest_sentinel_audit_matches_current` from
  `env_sentinel_audit_matches_current`. This matters because a copied shell
  command can export a stale
  `OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON` path while a newer audit
  artifact exists on disk. The previous queue logic could borrow freshness from
  the latest audit even when the env path itself would fail
  `operator_authorize.py`.
- `immediately_runnable_paid_launch=true` now requires the env-provided audit
  artifact to match current sentinel bytes. The latest audit is still used to
  construct a fresh `audit_backed_paid_launch_command`, but not to validate a
  stale operator environment.
- Regression coverage: stale env audit path rejected; current env audit
  snapshot accepted.

Follow-up snapshot-blocker refinement:

- The dirty-tree bypass now also requires
  `current_sentinel_snapshot_valid=true`. A missing/unreadable/outside-mount
  sentinel snapshot is no longer allowed to look runnable just because no dirty
  sentinel count could be computed.
- New precondition:
  `CATALOG_202_current_sentinel_snapshot_required_before_paid_dispatch`.
- Regression coverage: dirty tree + paired env + invalid current sentinel
  snapshot now fails closed before paid launch.

This is launch-custody hardening only. It does not change Candidate 4c's
predicted band, does not claim score movement, does not open a lane claim, and
does not launch Modal.

## 2026-05-18 Codex Candidate 4c Launch Packet Artifact-Dereference Repair

Adversarial launch-packet review found a stale-summary false-authority gap in
`tools/verify_candidate4c_launch_packet.py`: the doctor trusted the
queue-embedded `local_identity_disambiguator_probe` row without opening the
referenced `.omx/research/z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.json`
artifact. A stale or fabricated queue row could therefore carry
`runtime_output_changed=true` plus custody hashes and make the no-spend packet
look ready even if the actual probe artifact no longer matched.

Repair:

- `_local_identity_disambiguator_probe_status()` now resolves the queue-carried
  probe path under the current repo root and requires the JSON artifact to
  exist and parse;
- the artifact's verdict must match the queue row;
- the artifact's nested `inflate_output_comparison.runtime_output_changed` must
  be `true`;
- runtime custody aggregate SHA, full-output aggregate SHA, identity-output
  aggregate SHA, output root, and total byte differences must match the queue
  custody summary exactly;
- a regression test mutates the artifact after queue generation and proves the
  launch packet fails closed with artifact mismatch blockers.

Verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_verify_candidate4c_launch_packet.py -q
# 6 passed in 0.16s

.venv/bin/python tools/verify_candidate4c_launch_packet.py --json \
  | jq '{ready_for_operator_paid_execution, local_identity_disambiguator_probe_ready, local_identity_disambiguator_probe_blockers, result_review_blockers, active_lane_claims_clean, checks_ok}'
# ready_for_operator_paid_execution=true
# local_identity_disambiguator_probe_ready=true
# local_identity_disambiguator_probe_blockers=[]
# result_review_blockers=[]
# active_lane_claims_clean=true
# checks_ok=true
```

No provider job was launched and no lane claim was opened. This is launch
custody hardening only: Candidate 4c remains a no-spend operator-ready packet,
not a score claim or promotion claim.

## 2026-05-18 Codex Candidate 4c Paid-Command Audit Binding Gate

Follow-up review found a handoff-specific false-authority edge in the no-spend
packet. The packet's dry-run checks built a fresh environment from the current
Catalog #202 audit, but the operator-facing `next_paid_command` was copied from
the queue. If that queue command contained an older audit path or sentinel hash,
the packet could still look green while handing the operator a stale command.

Repair:

- `tools/verify_candidate4c_launch_packet.py` now verifies that the copied paid
  command contains:
  - `OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1`
  - `OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=catalog202_sentinel_audit:<current sentinel_set_sha256>`
  - `OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON=<current audit path>`
  - `OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1`
  - `OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=13.000`
  - `tools/run_modal_smoke_before_full.py`
  - the Candidate 4c recipe and operator handle
  - no `--dry-run`
- The packet now emits `next_paid_command_ready` and
  `next_paid_command_blockers`.
- Regression coverage injects a stale audit path/hash into the queue command
  and verifies launch readiness fails closed.

Fresh packet:
`.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T073219Z.json`

Packet verdict:

- `ready_for_operator_paid_execution=true`
- `local_identity_disambiguator_probe_ready=true`
- `next_paid_command_ready=true`
- `next_paid_command_blockers=[]`
- `provider_dispatch_attempted=false`
- `lane_claim_opened=false`
- `score_claim=false`
- `promotion_eligible=false`

Verification:

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_verify_candidate4c_launch_packet.py
# 5 passed

.venv/bin/python -m py_compile tools/verify_candidate4c_launch_packet.py
# PASS
```

No provider job was launched and no lane claim was opened.

## 2026-05-18 Codex Candidate 4c Launch-Packet Custody Gate

Adversarial review found one more stale-queue false-authority edge: after the
queue learned to carry `local_identity_disambiguator_probe.custody`, the
no-spend launch packet copied that object but did not require it before
returning `ready_for_operator_paid_execution=true`. An older queue artifact
could therefore have looked launch-ready while lacking the runtime/output hashes
that make the local no-op proof auditable.

Repair:

- `tools/verify_candidate4c_launch_packet.py` now requires the queue-carried
  local disambiguator proof to have:
  - `verdict=pending_paired_exact_eval_json`
  - `runtime_output_changed=true`
  - empty probe blockers
  - runtime custody aggregate SHA-256
  - full and identity raw-output aggregate SHA-256s
  - positive raw-output byte-difference count
  - output root
- The launch packet now emits
  `local_identity_disambiguator_probe_ready` and
  `local_identity_disambiguator_probe_blockers`.
- Regression coverage mutates the fixture into an old-style incomplete custody
  block and verifies the packet refuses readiness.

Fresh packet:
`.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T072757Z.json`

Packet verdict:

- `ready_for_operator_paid_execution=true`
- `local_identity_disambiguator_probe_ready=true`
- `local_identity_disambiguator_probe_blockers=[]`
- `checks_ok=true`
- `active_lane_claims_clean=true`
- `provider_dispatch_attempted=false`
- `lane_claim_opened=false`
- `score_claim=false`
- `promotion_eligible=false`

Verification:

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_verify_candidate4c_launch_packet.py
# 4 passed

.venv/bin/python -m py_compile tools/verify_candidate4c_launch_packet.py
# PASS
```

No provider job was launched and no lane claim was opened.

## 2026-05-18 Codex Candidate 4c Runtime-Custodied Inflate-Output Proof

Follow-up custody review found the local full-vs-identity disambiguator had
raw-output hashes but no top-level runtime-tree hash. That left the proof
usable for no-op detection, but weaker than the AGENTS apples-to-apples
mechanism packet requirement: archive SHA, runtime content SHA, and inflated
raw-output aggregate SHA must travel together.

Repair:

- `tools/probe_z6_predictive_coding_vs_identity_disambiguator.py` now hashes the
  inflate runtime closure used by the local comparison: `inflate.sh`, sibling
  `inflate.py`, and bundled `src/` source files.
- Archive payloads are deliberately excluded from the runtime hash and remain
  tracked under `source_archives`; Python cache files are also excluded so the
  runtime digest is source-stable.
- The regenerated disambiguator artifact carries the same custody object at
  top level and under `inflate_output_comparison.runtime_custody`.

Updated artifact:
`.omx/research/z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.json`

Current local advisory evidence:

- evidence axis: `[local-inflate-output advisory]`
- score authority: `false`
- promotion eligible: `false`
- runtime custody aggregate SHA-256:
  `384938f3b6a14acb5944938c45c127ccc411f00983aff7589c7c9b98cfc56073`
- runtime closure file count: `10`
- archive payloads excluded from runtime hash:
  `0.bin`, `0_identity_predictor_disambiguator.bin`, `archive.zip`,
  `archive_identity_predictor_disambiguator.zip`
- Python cache files excluded from runtime hash: `13`
- fresh output root:
  `experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z/z6_identity_inflate_output_comparison_runtime_source_custody_compact_20260518_codex`
- full output aggregate SHA-256:
  `2b0e3345c5eb8f00beb71de62e0bdda60cc17933acbbe775ef451b2791aacf73`
- identity output aggregate SHA-256:
  `856f15ea620ff704a3bebcdfef75f289e7af11d0d1e7b45f0fee7262505c6409`
- total raw-output byte differences: `22253`
- result: `runtime_output_changed=true`

Verification:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_probe_z6_predictive_coding_vs_identity_disambiguator.py
# 14 passed

.venv/bin/python -m py_compile \
  tools/probe_z6_predictive_coding_vs_identity_disambiguator.py
# PASS

.venv/bin/python tools/probe_z6_predictive_coding_vs_identity_disambiguator.py \
  --run-dir experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z \
  --inflate-sh experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z/submission_dir/inflate.sh \
  --file-list upstream/public_test_video_names.txt \
  --inflate-output-root experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z/z6_identity_inflate_output_comparison_runtime_source_custody_compact_20260518_codex \
  --inflate-python .venv/bin/python \
  --output-json .omx/research/z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.json \
  --output-md .omx/research/z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.md
# verdict=pending_paired_exact_eval_json; score_claim=false; promotion_eligible=false
```

No provider job was launched and no lane claim was opened. The next
authoritative gate remains paired contest-CUDA exact eval with matching ZIP and
runtime custody.

## 2026-05-18 Codex Candidate 4c Queue/Packet Runtime-Custody Wire-In

The runtime-custodied local disambiguator proof is now propagated into the
Race Mode queue and Candidate 4c no-spend launch packet, not just the
standalone probe artifact.

Updated machine-readable artifacts:

- refreshed Catalog #202 sentinel audit:
  `.omx/state/catalog202_sentinel_cleanliness/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch_20260518T072226Z.json`
- refreshed dispatch queue:
  `.omx/state/asymptotic_pursuit/dispatch_queue_20260518T072231Z.json`
- refreshed no-spend launch packet:
  `.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T072242Z.json`

The first launch-packet regeneration intentionally failed closed because the
previous Catalog #202 audit referenced the old
`tools/probe_z6_predictive_coding_vs_identity_disambiguator.py` hash. The new
audit records:

- `sentinel_set_sha256=a04c09b40cca80dd98c41968770ce7d2b19b672e2553d6a2bbb5c03b1a5aa387`
- dirty sentinels:
  - `tools/operator_authorize.py`
  - `tools/probe_z6_predictive_coding_vs_identity_disambiguator.py`
  - `tools/run_modal_smoke_before_full.py`
- `ready_for_catalog202_audit_backed_dirty_sentinel_attestation=true`

The refreshed queue and packet now carry
`local_identity_disambiguator_probe.custody` with:

- runtime custody aggregate SHA-256:
  `384938f3b6a14acb5944938c45c127ccc411f00983aff7589c7c9b98cfc56073`
- full output aggregate SHA-256:
  `2b0e3345c5eb8f00beb71de62e0bdda60cc17933acbbe775ef451b2791aacf73`
- identity output aggregate SHA-256:
  `856f15ea620ff704a3bebcdfef75f289e7af11d0d1e7b45f0fee7262505c6409`
- total raw-output byte differences: `22253`

Final packet verdict:

- `ready_for_operator_paid_execution=true`
- `active_lane_claims_clean=true`
- `checks_ok=true`
- `provider_dispatch_attempted=false`
- `lane_claim_opened=false`
- `score_claim=false`
- `promotion_eligible=false`

No provider job was launched and no lane claim was opened.

## 2026-05-18 Codex Candidate 4c No-Spend Launch Packet

The audit-backed command path now has a reusable no-spend launch packet doctor:
`tools/verify_candidate4c_launch_packet.py`.

Artifact:
`.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T070956Z.json`

Inputs chained by the packet:

- Queue:
  `.omx/state/asymptotic_pursuit/dispatch_queue_20260518T070955Z.json`
- Catalog #202 sentinel audit:
  `.omx/state/catalog202_sentinel_cleanliness/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch_20260518T065922Z.json`

No-spend checks inside the packet:

- `lane_claim_summary`: `returncode=0`, `active=0`
- `required_input_validation`: `returncode=0`
- `smoke_before_full_dry_run`: `returncode=0`
- `operator_authorize_dry_run`: `returncode=0`
- `catalog202_audit_backed_bypass_probe`: `returncode=0`

Packet verdict:

- `ready_for_operator_paid_execution=true`
- `active_lane_claims_clean=true`
- `checks_ok=true`
- `provider_dispatch_attempted=false`
- `lane_claim_opened=false`
- `score_claim=false`
- `promotion_eligible=false`
- `operator_action_required=true`

Current public frontier refresh in this pass found no new PR movement: latest
public PR remains #108, updated `2026-05-11T19:19:58Z`. No public-frontier
intake supersedes Candidate 4c.

Verification:

```bash
.venv/bin/python tools/verify_candidate4c_launch_packet.py --json --write-artifact \
  > /tmp/candidate4c_launch_packet.json
# rc=0; wrote .omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T070956Z.json

jq '{ready_for_operator_paid_execution, active_lane_claims_clean, checks_ok, provider_dispatch_attempted, lane_claim_opened, score_claim, promotion_eligible}' \
  /tmp/candidate4c_launch_packet.json
# ready_for_operator_paid_execution=true
# active_lane_claims_clean=true
# checks_ok=true
# provider_dispatch_attempted=false
# lane_claim_opened=false
# score_claim=false
# promotion_eligible=false
```

This is the clean pre-spend handoff state. The next action that can move score
is intentional operator execution of the audit-backed Candidate 4c paid command
or an explicit decision to defer spend and switch to a different byte-closed
candidate.

## 2026-05-18 Codex Audit-Backed Catalog #202 Launch Path

Supersession note for the prior `065035Z` sentinel audit: after wiring the
authorization verifier, `tools/operator_authorize.py` itself became an
effective dirty sentinel. The fresh audit is:
`.omx/state/catalog202_sentinel_cleanliness/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch_20260518T065922Z.json`.

New contract:

- Paired Catalog #202 env vars alone are still acceptable only when effective
  Modal sentinels are not dirty.
- If any effective Modal sentinel is dirty, `tools/operator_authorize.py` now
  requires `OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON=<audit.json>`.
- The audit JSON must match the current recipe name, effective sentinel file
  list, per-file SHA-256s, and aggregate `sentinel_set_sha256`.
- A missing or stale audit exits before `--require-clean-head` can be skipped.

Fresh Candidate 4c audit result:

- `sentinel_set_clean_for_catalog202=false`
- `sentinel_set_snapshot_stable_for_catalog202=true`
- `ready_for_catalog202_audit_backed_dirty_sentinel_attestation=true`
- `audit_backed_attestation_blockers=[]`
- dirty sentinel paths:
  - `tools/operator_authorize.py`
  - `tools/probe_z6_predictive_coding_vs_identity_disambiguator.py`
  - `tools/run_modal_smoke_before_full.py`
- `sentinel_set_sha256=9e94588c666d63cb187a766f71c6aaa75448efe35087a83e686e1800d04f1146`

Fresh queue artifact:
`.omx/state/asymptotic_pursuit/dispatch_queue_20260518T070955Z.json`.

The queue now emits an audit-backed paid command prefix:

```bash
OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1 \
OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=catalog202_sentinel_audit:9e94588c666d63cb187a766f71c6aaa75448efe35087a83e686e1800d04f1146 \
OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON=.omx/state/catalog202_sentinel_cleanliness/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch_20260518T065922Z.json \
OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=13.000 \
.venv/bin/python tools/run_modal_smoke_before_full.py \
  --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch \
  --operator-handle codex:z6_v2_candidate_4c_scorer_logit
```

No provider job was launched and no lane claim was opened. The command above is
a launch surface, not a score claim; it still requires intentional operator
execution.

Verification:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py \
  src/tac/tests/test_check_202_operator_authorize_clean_bypass.py \
  src/tac/tests/test_audit_catalog202_sentinel_cleanliness.py -q
# 75 passed

OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1 \
OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=operator-attests \
OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON=.omx/state/catalog202_sentinel_cleanliness/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch_20260518T065922Z.json \
.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, 'tools')
import operator_authorize as oa
recipe = oa._load_recipe('substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch')
print(oa._whole_tree_clean_check_bypass_active(recipe))
PY
# True, after verifying sentinel_set_sha256=9e94588c666d63cb187a766f71c6aaa75448efe35087a83e686e1800d04f1146

OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1 \
OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=operator-attests \
env -u OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON \
.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, 'tools')
import operator_authorize as oa
recipe = oa._load_recipe('substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch')
print(oa._whole_tree_clean_check_bypass_active(recipe))
PY
# exits 12 before bypass because dirty sentinels require the audit JSON
```

## 2026-05-18 Codex Catalog #202 Sentinel-Clean Audit

The dirty-tree queue precondition was sharpened into a byte-level audit helper:
`tools/audit_catalog202_sentinel_cleanliness.py`.

Purpose:

- Derive the effective Modal sentinel file set for a recipe using the same
  sentinel-selection rules as `tools/operator_authorize.py`.
- Hash each effective sentinel file.
- Compare sentinels against `git status --porcelain`.
- Emit a no-spend artifact proving whether the Catalog #202 paired-env
  attestation is currently defensible.

Live Candidate 4c audit artifact:
`.omx/state/catalog202_sentinel_cleanliness/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch_20260518T065035Z.json`

Result:

- `sentinel_set_clean_for_catalog202=false`
- `ready_for_catalog202_paired_env_attestation=false`
- `attestation_blockers=["catalog202_sentinel_files_dirty_in_git"]`
- `dirty_worktree_path_count=48`
- `dirty_sentinel_path_count=2`
- dirty sentinel paths:
  - `tools/probe_z6_predictive_coding_vs_identity_disambiguator.py`
  - `tools/run_modal_smoke_before_full.py`
- `recipe_git_status=" M"`
- `sentinel_set_sha256=73f59a16f74f12761cb6826e321d6d864277bb53de8f8f2fa5e2f055c46c1ed0`

Interpretation:

Candidate 4c remains the top logical paid-ready substrate, but the current
shared worktree cannot honestly use the Catalog #202 dirty-tree bypass yet.
This is stronger than "operator forgot env vars": the sentinel set itself is
dirty. Before paid provider dispatch, either the dirty sentinel changes must be
stabilized into the source-of-truth worktree or the recipe sentinel set must be
changed with an explicit rationale. The worker-side Catalog #166 hash check
would still compare local sentinels against mounted worker bytes, but Catalog
#202's intended "unrelated dirty files only" bypass condition is not satisfied.

Verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_audit_catalog202_sentinel_cleanliness.py -q
# 4 passed in 0.18s

.venv/bin/python -m py_compile tools/audit_catalog202_sentinel_cleanliness.py
# PASS

.venv/bin/python tools/audit_catalog202_sentinel_cleanliness.py \
  --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch \
  --json --write-artifact > /tmp/catalog202_candidate4c_audit.json
# rc=1; wrote .omx/state/catalog202_sentinel_cleanliness/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch_20260518T065035Z.json

jq '{sentinel_set_clean_for_catalog202, ready_for_catalog202_paired_env_attestation, attestation_blockers, dirty_sentinel_paths}' \
  /tmp/catalog202_candidate4c_audit.json
# sentinel_set_clean_for_catalog202=false
# ready_for_catalog202_paired_env_attestation=false
# attestation_blockers=["catalog202_sentinel_files_dirty_in_git"]
# dirty_sentinel_paths=[
#   "tools/probe_z6_predictive_coding_vs_identity_disambiguator.py",
#   "tools/run_modal_smoke_before_full.py"
# ]
```

No provider job was launched and no lane claim was opened.

## 2026-05-18 Codex Dirty-Tree Paid-Launch Precondition Repair

Follow-up adversarial review found a second paid-launch false-authority gap in
the Race Mode queue. Candidate 4c is logically READY, but the current shared
worktree is dirty. `tools/run_modal_smoke_before_full.py` deliberately does not
fabricate the Catalog #202 clean-head bypass; `tools/operator_authorize.py`
skips `--require-clean-head` only when the operator externally supplies both:

- `OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1`
- `OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=<operator verified sentinel-clean attestation>`

The queue now separates substrate readiness from immediate launchability:

- `ready_for_paid_dispatch_count=1`
- `immediately_runnable_paid_dispatch_count=0`
- `current_worktree_dirty_path_count=48`
- `top_ready_substrate=z6_v2_candidate_4c_scorer_logit`
- `top_ready_paid_launch_missing_preconditions=[
  "CATALOG_202_dirty_worktree_requires_paired_env_attestation_before_paid_dispatch"
  ]`
- `top_immediately_runnable_paid_launch_command=null`

Current machine artifact:
`.omx/state/asymptotic_pursuit/dispatch_queue_20260518T065153Z.json`

This preserves the important signal: Candidate 4c remains the top paid-ready
campaign candidate, but the copied paid command is not advertised as
immediately runnable while partner/WIP dirt is present and the Catalog #202
paired attestation has not been supplied.

Verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py -q
# 56 passed in 2.54s

.venv/bin/python -m py_compile \
  tools/asymptotic_pursuit_dispatch_queue.py \
  tools/asymptotic_pursuit_candidate_readiness_assessment.py
# PASS

.venv/bin/python tools/asymptotic_pursuit_dispatch_queue.py --json --write-artifact \
  > /tmp/pact_dispatch_queue_current.json
# wrote .omx/state/asymptotic_pursuit/dispatch_queue_20260518T065153Z.json

jq '{ready_for_paid_dispatch_count, immediately_runnable_paid_dispatch_count, current_worktree_dirty_path_count, top_ready_paid_launch_missing_preconditions, top_immediately_runnable_paid_launch_command}' \
  /tmp/pact_dispatch_queue_current.json
# ready_for_paid_dispatch_count=1
# immediately_runnable_paid_dispatch_count=0
# current_worktree_dirty_path_count=48
# top_ready_paid_launch_missing_preconditions=[
#   "CATALOG_202_dirty_worktree_requires_paired_env_attestation_before_paid_dispatch"
# ]
# top_immediately_runnable_paid_launch_command=null
```

No provider job was launched and no lane claim was opened.

## 2026-05-18 Codex Race-Mode Dispatch Queue Actuator Fields

Because `.omx/state/RACE_MODE_ACTIVE.flag` is present, I refreshed the
Candidate 4c queue handoff so the launchable state is machine-readable from
the dispatch queue itself, not only from terminal output or prose.

Code change:

- `tools/asymptotic_pursuit_dispatch_queue.py` now emits per-row
  `ready_for_paid_dispatch`, `dry_run_command`, `paid_launch_command`, and
  `operator_session_authorization`.
- The top-level payload now carries `ready_for_paid_dispatch_count`,
  `top_ready_substrate`, `top_ready_paid_launch_command`, and
  `top_ready_dry_run_command`.
- `src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py` pins the
  Candidate 4c queue row so the paid command keeps the `$13.000` session-budget
  floor while the dry-run command stays non-spend and env-free.

Fresh artifacts:

- readiness:
  `.omx/state/asymptotic_pursuit/readiness_assessment_20260518T063400Z.json`
- dispatch queue:
  `.omx/state/asymptotic_pursuit/dispatch_queue_20260518T063400Z.json`

Snapshot facts:

```json
{
  "top_1_substrate": "z6_v2_candidate_4c_scorer_logit",
  "top_1_readiness_verdict": "READY",
  "ready_for_paid_dispatch_count": 1,
  "top_ready_substrate": "z6_v2_candidate_4c_scorer_logit",
  "ready_total_cost_usd_if_dispatched": 2.083,
  "ready_total_session_budget_floor_usd": 13.0,
  "candidate4c": {
    "ready_for_paid_dispatch": true,
    "blocking_issues": [],
    "minimum_session_budget_usd": 13.0
  }
}
```

Dry-run actuator command, verified no-spend:

```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
  --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch \
  --operator-handle codex:z6_candidate4c_scorer_logit \
  --dry-run
# no Modal dispatch; would dispatch 100-epoch T4 smoke, then FULL only after SMOKE GREEN
```

Paid actuator command emitted by the queue, still requiring explicit operator
session budget:

```bash
OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=13.000 \
.venv/bin/python tools/run_modal_smoke_before_full.py \
  --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch \
  --operator-handle codex:z6_candidate4c_scorer_logit
```

Negative authorization proof:

```bash
env -u OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE \
  -u OPERATOR_AUTHORIZE_SESSION_BUDGET_USD \
  .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch \
    --operator-handle codex:z6_candidate4c_scorer_logit \
    --smoke-only
# exits 9 before claim/provider work:
# FATAL: paid session authorization missing ...
```

Input-custody proof:

```bash
.venv/bin/python tools/validate_dispatch_required_inputs.py \
  --trainer experiments/train_substrate_time_traveler_l5_z6.py \
  --flag-value=--video-path=upstream/videos/0.mkv
# OK: --video-path -> /Users/adpena/Projects/pact/upstream/videos/0.mkv
```

Verification:

```bash
.venv/bin/python -m py_compile \
  tools/asymptotic_pursuit_dispatch_queue.py \
  tools/asymptotic_pursuit_candidate_readiness_assessment.py
# PASS

.venv/bin/python -m pytest src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py -q
# 54 passed in 1.74s

jq empty \
  .omx/state/asymptotic_pursuit/readiness_assessment_20260518T063400Z.json \
  .omx/state/asymptotic_pursuit/dispatch_queue_20260518T063400Z.json
# PASS
```

No provider job was launched and no lane claim was opened.

## 2026-05-18 Codex Dispatch Queue Snapshot Artifact

The Candidate 4c readiness path now has a durable queue snapshot writer instead
of relying on transient `--json` terminal output. This closes a no-signal-loss
handoff gap: the operator or a later agent can recover the exact TOP-1 queue,
cost rollup, local disambiguator guard, and false-authority blockers from disk.

Code change:

- `tools/asymptotic_pursuit_dispatch_queue.py` now supports
  `--write-artifact` and writes a timestamped JSON payload under
  `.omx/state/asymptotic_pursuit/`.
- The payload includes `result_review_blockers` so the queue is clearly a
  planning artifact, not score or promotion authority.
- The Candidate 4c row carries `local_identity_disambiguator_probe.path`,
  `verdict`, `runtime_output_changed`, and `blockers`.

Fresh artifacts:

- readiness snapshot:
  `.omx/state/asymptotic_pursuit/readiness_assessment_20260518T051806Z.json`
- dispatch queue snapshot:
  `.omx/state/asymptotic_pursuit/dispatch_queue_20260518T051806Z.json`

Snapshot summary:

```json
{
  "top_1_substrate": "z6_v2_candidate_4c_scorer_logit",
  "top_1_readiness_verdict": "READY",
  "ready_count": 1,
  "ready_total_cost_usd_if_dispatched": 2.083,
  "ready_total_session_budget_floor_usd": 13.0,
  "candidate4c": {
    "readiness_verdict": "READY",
    "blocking_issues": [],
    "local_identity_disambiguator_probe": {
      "verdict": "pending_paired_exact_eval_json",
      "runtime_output_changed": true,
      "blockers": []
    }
  }
}
```

Verification:

```bash
.venv/bin/python -m py_compile \
  tools/asymptotic_pursuit_dispatch_queue.py \
  tools/asymptotic_pursuit_candidate_readiness_assessment.py
# PASS

.venv/bin/python -m pytest src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py -q
# 50 passed in 1.38s

.venv/bin/python tools/asymptotic_pursuit_candidate_readiness_assessment.py \
  --write-artifact --json
# wrote readiness_assessment_20260518T051806Z.json; TOP-1 READY Candidate 4c

.venv/bin/python tools/asymptotic_pursuit_dispatch_queue.py --write-artifact --json
# wrote dispatch_queue_20260518T051806Z.json; READY queue estimate $2.083,
# operator session budget floor $13.0, local disambiguator blockers []
```

No provider job was launched and no lane claim was opened.

## 2026-05-18 Codex Candidate 4c Identity Archive-Pair Output Probe

I upgraded `tools/probe_z6_predictive_coding_vs_identity_disambiguator.py` so
archive-pair mode can optionally run the same `inflate.sh` against staged
full-FiLM and identity-predictor archives, hash both raw-output trees, and
fail closed if the predictor switch is a runtime no-op. This keeps the local
output evidence on the canonical Z6 disambiguator surface rather than adding a
one-off comparer.

Artifact:

- JSON:
  `.omx/research/z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.json`
- Markdown:
  `.omx/research/z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.md`
- full archive ZIP: `194119` bytes,
  SHA-256 `47a776cd36577f2cd6026b71c52187618839e9fae70e30f5d8c7eaf8b1047855`
- identity archive ZIP: `194295` bytes,
  SHA-256 `a2c47579f2d7e1e0c7b950e7ca7deca8fb0d8d414fd17a49d4b7946e19479d64`
- identity ZIP byte overhead: `176` bytes
- identity rate-term overhead:
  `0.00011719117574950216`
- shared archive checks: encoder/decoder/predictor state dicts, latent init,
  residuals, and ego-motion all match across arms
- local inflate output comparison axis: `[local-inflate-output advisory]`
- full output aggregate SHA-256:
  `2b0e3345c5eb8f00beb71de62e0bdda60cc17933acbbe775ef451b2791aacf73`
- identity output aggregate SHA-256:
  `856f15ea620ff704a3bebcdfef75f289e7af11d0d1e7b45f0fee7262505c6409`
- raw output bytes per arm: `12208032`
- raw output file set: identical (`0.raw`)
- raw output byte differences: `22253`
- result: `runtime_output_changed=true`; the identity-predictor flag is
  consumed by the vendored inflate runtime and is not a local no-op

Status remains deliberately fail-closed:

- verdict: `pending_paired_exact_eval_json`
- evidence grade: `byte_closed_archive_pair_no_score`
- blockers: `no_paired_exact_eval_json`, `no_contest_cpu_cuda_pair`,
  `not_score_authority`
- `research_only=true`, `score_claim=false`, `promotion_eligible=false`,
  `ready_for_exact_eval_dispatch=false`

Command:

```bash
.venv/bin/python tools/probe_z6_predictive_coding_vs_identity_disambiguator.py \
  --run-dir experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z \
  --inflate-sh experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z/submission_dir/inflate.sh \
  --file-list upstream/public_test_video_names.txt \
  --inflate-output-root experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z/z6_identity_inflate_output_comparison_20260518_codex \
  --inflate-python .venv/bin/python \
  --output-json .omx/research/z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.json \
  --output-md .omx/research/z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.md
# verdict=pending_paired_exact_eval_json evidence_grade=byte_closed_archive_pair_no_score
```

Verification:

```bash
.venv/bin/python -m py_compile tools/probe_z6_predictive_coding_vs_identity_disambiguator.py
# PASS

.venv/bin/python -m pytest src/tac/tests/test_probe_z6_predictive_coding_vs_identity_disambiguator.py -q
# 14 passed in 1.80s
```

No provider job was launched and no lane claim was opened.

## 2026-05-18 Codex Candidate 4c Readiness Wire-In For Local Disambiguator Proof

Follow-up adversarial review found an orphan-signal risk: the Candidate 4c
recipe already said its planning band was blocked from promotion/rank authority
until the full-vs-identity disambiguator and post-training validation landed,
but the asymptotic readiness path did not consume the newly emitted local
archive-pair/output-comparison proof. That meant the artifact could sit as a
side report while the paid command stayed READY even if a future local probe
proved the identity switch was a runtime no-op.

Repair:

- `tools/probe_z6_predictive_coding_vs_identity_disambiguator.py` now emits
  `research_only=true` on probe payloads and nested local inflate-output
  comparisons.
- Candidate 4c's recipe now names the local disambiguator artifact:
  `.omx/research/z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.json`.
- `tools/asymptotic_pursuit_candidate_readiness_assessment.py` validates the
  recipe-wired probe before preserving READY status. It blocks paid readiness
  if the probe is missing, outside the repo, malformed, score-authoritative,
  missing false-authority flags, missing local inflate-output comparison, or
  records `runtime_output_changed=false`.
- `tools/asymptotic_pursuit_dispatch_queue.py` carries the probe path, verdict,
  runtime-output-changed bit, and blockers into the ordered queue.

Live Candidate 4c state after wire-in:

```json
{
  "top_1_substrate": "z6_v2_candidate_4c_scorer_logit",
  "top_1_readiness_verdict": "READY",
  "candidate4c": {
    "readiness_verdict": "READY",
    "blocking_issues": [],
    "local_identity_disambiguator_probe_verdict": "pending_paired_exact_eval_json",
    "local_identity_disambiguator_runtime_output_changed": true,
    "local_identity_disambiguator_blockers": []
  }
}
```

Dispatch queue rollup remains unchanged on spend math, but now includes the
local no-op guard artifact:

- READY count: `1`
- READY queue estimate: `$2.083`
- READY operator session budget floor: `$13.0`
- Candidate 4c local disambiguator verdict:
  `pending_paired_exact_eval_json`
- Candidate 4c local runtime-output-changed: `true`
- Candidate 4c local disambiguator blockers: `[]`

Verification:

```bash
.venv/bin/python -m py_compile \
  tools/probe_z6_predictive_coding_vs_identity_disambiguator.py \
  tools/asymptotic_pursuit_candidate_readiness_assessment.py \
  tools/asymptotic_pursuit_dispatch_queue.py
# PASS

.venv/bin/python -m pytest \
  src/tac/tests/test_probe_z6_predictive_coding_vs_identity_disambiguator.py \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py -q
# 63 passed in 3.10s

.venv/bin/python tools/asymptotic_pursuit_candidate_readiness_assessment.py --json \
  | jq '{top_1_substrate, top_1_readiness_verdict, candidate4c:([.candidates[] | select(.substrate_id=="z6_v2_candidate_4c_scorer_logit") | {readiness_verdict, blocking_issues, local_identity_disambiguator_probe_verdict, local_identity_disambiguator_runtime_output_changed, local_identity_disambiguator_blockers}] | .[0])}'
# top_1_substrate=z6_v2_candidate_4c_scorer_logit
# top_1_readiness_verdict=READY
# candidate4c.blocking_issues=[]
# candidate4c.local_identity_disambiguator_runtime_output_changed=true

.venv/bin/python tools/run_modal_smoke_before_full.py \
  --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch \
  --operator-handle codex:z6_v2_candidate_4c_scorer_logit \
  --dry-run
# --dry-run; no Modal dispatch; would dispatch SMOKE then FULL only after SMOKE GREEN.

.venv/bin/python tools/local_pre_deploy_check.py \
  --trainer experiments/train_substrate_time_traveler_l5_z6.py \
  --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch \
  --strict
# ALL 9 CHECKS PASSED.
```

No provider job was launched and no lane claim was opened.

## 2026-05-18 Codex Operator Command Path Repair

The readiness assessment's generated TOP-1 operator command was not directly
runnable: it emitted
`--recipe /Users/adpena/Projects/pact/.omx/operator_authorize_recipes/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch.yaml`,
but `tools/run_modal_smoke_before_full.py` resolves `--recipe` as a recipe name
under `.omx/operator_authorize_recipes/`. Passing a path would therefore resolve
to a doubled recipe directory and fail before the smoke could even dry-run.

`tools/asymptotic_pursuit_candidate_readiness_assessment.py` now emits the
recipe basename plus an explicit operator handle:

```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
  --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch \
  --operator-handle codex:z6_v2_candidate_4c_scorer_logit
```

This keeps the TOP-1 READY recommendation attached to the actual
smoke-before-full actuator surface. It is command-surface hardening only:
no provider job was launched and no lane claim was opened.

Verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py -q
# 44 passed in 1.00s

.venv/bin/python tools/asymptotic_pursuit_candidate_readiness_assessment.py --json \
  | jq -r '.top_1_operator_authorize_command'
# .venv/bin/python tools/run_modal_smoke_before_full.py --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch --operator-handle codex:z6_v2_candidate_4c_scorer_logit ...

.venv/bin/python tools/run_modal_smoke_before_full.py \
  --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch \
  --operator-handle codex:z6_v2_candidate_4c_scorer_logit \
  --dry-run
# --dry-run; no Modal dispatch; recipe resolves and would smoke before full.
```

## 2026-05-18 Codex Smoke Wrapper Recipe-Path Tolerance

The generated TOP-1 command now uses a recipe basename, but older handoff lines
and historical ledgers still contain path-style invocations such as
`.omx/operator_authorize_recipes/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch.yaml`
or the absolute path under `/Users/adpena/Projects/pact`. Before this repair,
`tools/run_modal_smoke_before_full.py` prepended `.omx/operator_authorize_recipes`
to any argument ending in `.yaml`, so those valid-looking handoff commands
resolved to a doubled path and failed before the smoke dry-run.

`tools/run_modal_smoke_before_full.py` now accepts all three forms:

- `substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch`
- `substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch.yaml`
- `.omx/operator_authorize_recipes/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch.yaml`

Absolute recipe paths are accepted too. Dry-run output now normalizes the
planned dispatch recipe to the stem, matching the actual `_spawn_smoke_dispatch`
behavior.

Verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_run_modal_smoke_before_full.py -q
# 31 passed in 0.18s

.venv/bin/python tools/run_modal_smoke_before_full.py \
  --recipe .omx/operator_authorize_recipes/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch.yaml \
  --operator-handle codex:z6_v2_candidate_4c_scorer_logit \
  --dry-run
# --dry-run; recipe resolves; would dispatch SMOKE with recipe=substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch.

.venv/bin/python tools/run_modal_smoke_before_full.py \
  --recipe /Users/adpena/Projects/pact/.omx/operator_authorize_recipes/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch.yaml \
  --operator-handle codex:z6_v2_candidate_4c_scorer_logit \
  --dry-run
# --dry-run; recipe resolves; would dispatch SMOKE with recipe=substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch.
```

No provider job was launched and no lane claim was opened.

## 2026-05-18 Codex Paid Smoke Authorization Guard

Adversarial review found a launch-surface false-authority gap after the TOP-1
command-path repair: `tools/run_modal_smoke_before_full.py` invokes
`tools/operator_authorize.py --yes` internally, but the generated readiness
command did not include the paired session directive/budget env vars that the
Candidate 4c recipe's paid path already documents. A copied non-dry-run command
could therefore bypass the normal interactive confirmation path before the
wrapper reached Modal preflight, lane claim, or provider setup.

Repair:

- `tools/run_modal_smoke_before_full.py` now refuses every non-dry-run paid
  smoke/full invocation unless both
  `OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1` and a positive
  `OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=<float>` are present.
- Dry-runs remain free and still require no env vars.
- `tools/asymptotic_pursuit_candidate_readiness_assessment.py` now emits the
  paired env prefix directly in the TOP-1 command.
- Follow-up adversarial review found the queue estimate and recipe authorization
  envelope were being conflated: the queue's current smoke + full contest-CUDA
  + paired CPU estimate is `$2.083`, but the recipe declares a conservative
  `cost_band.predicted_cost_usd: 13.0` envelope and its operator notes require
  `$13.00`. The wrapper now rejects `OPERATOR_AUTHORIZE_SESSION_BUDGET_USD`
  below the recipe floor, and the live TOP-1 command carries
  `OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=13.000`.
- `tools/asymptotic_pursuit_dispatch_queue.py` now emits both
  `total_estimated_cost_usd` and `operator_session_budget_floor_usd` so
  downstream consumers cannot mistake an estimated spend rollup for the
  operator authorization envelope.

This is a dispatch-safety and no-silent-spend repair, not a score claim. It
does not open a lane claim, launch Modal, or alter Candidate 4c's READY status.

Verification:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_run_modal_smoke_before_full.py \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py -q
# 82 passed in 1.25s

.venv/bin/python -m py_compile \
  tools/run_modal_smoke_before_full.py \
  tools/asymptotic_pursuit_candidate_readiness_assessment.py \
  tools/asymptotic_pursuit_dispatch_queue.py
# PASS

env -u OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE \
  -u OPERATOR_AUTHORIZE_SESSION_BUDGET_USD \
  .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch \
    --operator-handle codex:z6_v2_candidate_4c_scorer_logit \
    --smoke-only
# exits 9 before claim/provider work:
# FATAL: paid session authorization missing ...

OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
  OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=2.083 \
  .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch \
    --operator-handle codex:z6_v2_candidate_4c_scorer_logit \
    --smoke-only
# exits 9 before claim/provider work:
# FATAL: ... below the recipe budget floor $13.000

.venv/bin/python tools/asymptotic_pursuit_candidate_readiness_assessment.py --json \
  | jq -r '.top_1_operator_authorize_command'
# OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1
# OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=13.000
# .venv/bin/python tools/run_modal_smoke_before_full.py ...

.venv/bin/python tools/asymptotic_pursuit_dispatch_queue.py --json \
  | jq '{ready_total_cost_usd_if_dispatched:.cost_band_rollup.ready_total_cost_usd_if_dispatched, ready_total_session_budget_floor_usd:.cost_band_rollup.ready_total_session_budget_floor_usd}'
# ready_total_cost_usd_if_dispatched=2.083
# ready_total_session_budget_floor_usd=13.0
```

No provider job was launched and no lane claim was opened.

## 2026-05-18 Codex Candidate 4c Queue-Immediate Launch-Packet Gate

Adversarial launch-packet review found a downstream false-authority bug after
the dispatch queue learned to distinguish `ready_for_paid_dispatch` from
`immediately_runnable_paid_launch`. The no-spend packet still reduced queue
readiness to `top_ready_substrate == Candidate 4c` plus an audit-backed command
string. A stale or copied queue artifact could therefore return
`ready_for_operator_paid_execution=true` even while the queue itself carried
`immediately_runnable_paid_dispatch_count=0` and non-empty Catalog #202 missing
preconditions.

Repair:

- `tools/verify_candidate4c_launch_packet.py` now requires the Candidate 4c
  dispatch row to be both `ready_for_paid_dispatch=true` and
  `immediately_runnable_paid_launch=true`.
- The packet requires empty top-level and row-level
  `paid_launch_missing_preconditions`, a positive
  `immediately_runnable_paid_dispatch_count`, and a non-empty
  `top_immediately_runnable_paid_launch_command`.
- Catalog #202 queue metadata must show the attestation environment satisfied,
  the env-provided sentinel audit matching current bytes, and a valid current
  sentinel snapshot when those checks are required.
- The packet now emits `queue_immediate_launch_ready` and
  `queue_immediate_launch_blockers` so a command can be distinguished from a
  launchable queue state.
- Regression coverage mutates the fixture into the exact false-authority shape:
  command present, but immediate-runnable count zero and Catalog #202
  preconditions missing.

Live latest-queue verdict after the repair:

```bash
zsh -o pipefail -c '.venv/bin/python tools/verify_candidate4c_launch_packet.py --json | jq "{ready_for_operator_paid_execution, queue_immediate_launch_ready, queue_immediate_launch_blockers, queue_immediately_runnable_paid_dispatch_count, queue_top_ready_paid_launch_missing_preconditions, next_paid_command_ready, active_lane_claims_clean, checks_ok, provider_dispatch_attempted, lane_claim_opened, score_claim, promotion_eligible}"'
# exits 1
# ready_for_operator_paid_execution=false
# queue_immediate_launch_ready=false
# queue_immediate_launch_blockers=[
#   candidate4c_queue_row_not_immediately_runnable,
#   candidate4c_queue_top_ready_paid_launch_preconditions_missing,
#   candidate4c_queue_row_paid_launch_preconditions_missing,
#   candidate4c_queue_no_immediately_runnable_paid_dispatch,
#   candidate4c_queue_top_immediately_runnable_command_missing,
#   candidate4c_queue_catalog202_env_attestation_not_satisfied,
#   candidate4c_queue_catalog202_env_sentinel_audit_not_current,
#   candidate4c_queue_catalog202_current_sentinel_snapshot_not_valid
# ]
# queue_immediately_runnable_paid_dispatch_count=0
# queue_top_ready_paid_launch_missing_preconditions=[
#   CATALOG_202_dirty_worktree_requires_paired_env_attestation_before_paid_dispatch
# ]
# next_paid_command_ready=true
# active_lane_claims_clean=true
# checks_ok=true
# provider_dispatch_attempted=false
# lane_claim_opened=false
# score_claim=false
# promotion_eligible=false
```

Verification:

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_verify_candidate4c_launch_packet.py
# 7 passed in 0.16s

.venv/bin/python -m py_compile \
  tools/verify_candidate4c_launch_packet.py \
  src/tac/tests/test_verify_candidate4c_launch_packet.py
# PASS

.venv/bin/python -m pytest -q \
  src/tac/tests/test_verify_candidate4c_launch_packet.py \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py \
  src/tac/tests/test_check_202_operator_authorize_clean_bypass.py \
  src/tac/tests/test_run_modal_smoke_before_full.py
# 118 passed in 14.84s

git diff --check
# PASS

.venv/bin/python tools/check_dispatch_cli_shell_hazards.py --strict
# PASS
```

No provider job was launched and no lane claim was opened.

### Follow-up: Catalog #202 Bypass Probe Semantic Gate

The same launch-packet review found a narrower subprocess false-authority bug:
`catalog202_audit_backed_bypass_probe` invoked
`operator_authorize._whole_tree_clean_check_bypass_active(recipe)` and treated
return code 0 as success. That Python probe can print `False` and still exit
0, so a semantically rejected Catalog #202 bypass could have appeared as an OK
check if the surrounding dry-run checks were stale, mocked, or changed.

Repair:

- `tools/verify_candidate4c_launch_packet.py` now requires the last non-empty
  probe stdout line to be exactly `True`;
- when the probe prints anything else, the check's `ok` field is forced false;
- the packet emits `catalog202_audit_backed_bypass_probe_accepted`;
- readiness gets the explicit blocker
  `candidate4c_catalog202_audit_backed_bypass_probe_not_true`;
- regression coverage proves `returncode=0, stdout=False` no longer blesses the
  packet.

Fresh latest packet after the semantic gate:
`.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T091134Z.json`

Live packet verdict:

- `ready_for_operator_paid_execution=false`
- `queue_immediate_launch_ready=false`
- `catalog202_audit_backed_bypass_probe_accepted=true`
- `checks_ok=true`
- `next_paid_command_ready=true`
- `provider_dispatch_attempted=false`
- `lane_claim_opened=false`
- `score_claim=false`
- `promotion_eligible=false`

Interpretation: older `ready_for_operator_paid_execution=true` packet artifacts
from before the queue-immediate gate are historical only. The newest packet is
fail-closed because the current shell does not satisfy the queue-immediate
Catalog #202 preconditions, even though the copied paid command is syntactically
audit-bound and the bypass probe itself accepts the current audit.

Verification:

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_verify_candidate4c_launch_packet.py
# 8 passed in 0.17s

.venv/bin/python -m py_compile \
  tools/verify_candidate4c_launch_packet.py \
  src/tac/tests/test_verify_candidate4c_launch_packet.py
# PASS

git diff --check -- \
  tools/verify_candidate4c_launch_packet.py \
  src/tac/tests/test_verify_candidate4c_launch_packet.py
# PASS

.venv/bin/python -m pytest -q \
  src/tac/tests/test_verify_candidate4c_launch_packet.py \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py \
  src/tac/tests/test_check_202_operator_authorize_clean_bypass.py \
  src/tac/tests/test_run_modal_smoke_before_full.py
# 119 passed in 14.90s

git diff --check
# PASS

.venv/bin/python tools/check_dispatch_cli_shell_hazards.py --strict
# PASS
```

No provider job was launched and no lane claim was opened.

## 2026-05-18 Codex Candidate 4c Env-Attested Immediate Launch Packet

After the queue-immediate launch-packet gate, the default shell correctly
failed closed because Catalog #202 env preconditions were not set. This pass
replayed the handoff with the exact operator-attested Catalog #202 env tuple
already named by the current sentinel audit:

```bash
export OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1
export OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=catalog202_sentinel_audit:a04c09b40cca80dd98c41968770ce7d2b19b672e2553d6a2bbb5c03b1a5aa387
export OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON=.omx/state/catalog202_sentinel_cleanliness/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch_20260518T072226Z.json
export OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1
export OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=13.000
```

Artifacts:

- env-attested queue:
  `.omx/state/asymptotic_pursuit/dispatch_queue_20260518T091517Z.json`
- env-attested no-spend packet:
  `.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T091519Z.json`

Queue verdict:

- `ready_for_paid_dispatch_count=1`
- `immediately_runnable_paid_dispatch_count=1`
- `top_ready_substrate=z6_v2_candidate_4c_scorer_logit`
- `top_ready_paid_launch_missing_preconditions=[]`
- Candidate 4c row:
  - `ready_for_paid_dispatch=true`
  - `immediately_runnable_paid_launch=true`
  - `paid_launch_missing_preconditions=[]`
  - `satisfied_in_current_environment=true`
  - `env_sentinel_audit_matches_current=true`
  - `current_sentinel_snapshot_valid=true`

Packet verdict:

- `ready_for_operator_paid_execution=true`
- `queue_immediate_launch_ready=true`
- `queue_immediate_launch_blockers=[]`
- `catalog202_audit_backed_bypass_probe_accepted=true`
- `checks_ok=true`
- `active_lane_claims_clean=true`
- `next_paid_command_ready=true`
- `provider_dispatch_attempted=false`
- `lane_claim_opened=false`
- `score_claim=false`
- `promotion_eligible=false`

Interpretation: Candidate 4c is not launchable from an unauthenticated shell,
but it is launch-packet-green when the operator supplies the Catalog #202
sentinel audit/hash env tuple and the `$13.000` session-budget directive. This
is still a no-spend artifact: no provider job was launched, no lane claim was
opened, and no score or promotion claim exists. The next score-moving action is
an explicit claimed paid smoke/full handoff from this env-attested packet.

Verification:

```bash
.venv/bin/python tools/verify_candidate4c_launch_packet.py --json \
  --queue-path .omx/state/asymptotic_pursuit/dispatch_queue_20260518T091517Z.json \
  | jq '{queue_path, ready_for_operator_paid_execution, queue_immediate_launch_ready, queue_immediate_launch_blockers, catalog202_audit_backed_bypass_probe_accepted, checks_ok, active_lane_claims_clean, next_paid_command_ready, provider_dispatch_attempted, lane_claim_opened, score_claim, promotion_eligible}'
# rc=0
# ready_for_operator_paid_execution=true
# queue_immediate_launch_ready=true
# queue_immediate_launch_blockers=[]
# catalog202_audit_backed_bypass_probe_accepted=true
# checks_ok=true
# active_lane_claims_clean=true
# next_paid_command_ready=true
# provider_dispatch_attempted=false
# lane_claim_opened=false
# score_claim=false
# promotion_eligible=false

.venv/bin/python -m pytest -q \
  src/tac/tests/test_verify_candidate4c_launch_packet.py \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py
# 69 passed in 12.33s

.venv/bin/python -m py_compile \
  tools/verify_candidate4c_launch_packet.py \
  src/tac/tests/test_verify_candidate4c_launch_packet.py \
  tools/asymptotic_pursuit_dispatch_queue.py
# PASS

git diff --check
# PASS

.venv/bin/python tools/check_dispatch_cli_shell_hazards.py --strict
# PASS

.venv/bin/python tools/claim_lane_dispatch.py summary
# CLAIM_SUMMARY active=0 stale_nonterminal=0 terminal_latest=942 ...
```

## 2026-05-18 Codex Candidate 4c Codex Pre-Dispatch Review Blocker

The env-attested queue/packet proved Candidate 4c can pass local no-spend
handoff checks, but the mandatory Catalog #271 codex pre-dispatch review found
a launch-blocking axis mismatch before any GPU spend:

- review artifact:
  `.omx/state/candidate4c_launch_packet/candidate4c_codex_pre_dispatch_review_20260518T0920Z.json`
- post-review packet:
  `.omx/state/candidate4c_launch_packet/candidate4c_no_spend_launch_packet_20260518T092725Z.json`

No-spend pre-dispatch facts:

- `tools/local_pre_deploy_check.py --strict`: PASS, 9/9 checks
- smoke-before-full dry-run: PASS, would dispatch `100` epochs on T4 and full
  only after smoke green
- codex pre-dispatch review: `verdict=needs-attention`, `rc=1`
- latest packet: `ready_for_operator_paid_execution=false`
- latest packet: `queue_immediate_launch_ready=true`
- latest packet:
  `codex_pre_dispatch_review_blockers=[
  candidate4c_codex_pre_dispatch_review_blocking_needs-attention]`
- `provider_dispatch_attempted=false`
- `lane_claim_opened=false`
- `score_claim=false`
- `promotion_eligible=false`

Blocking finding recovered from the codex review:

```text
- [high] Contest-CUDA smoke contract is incompatible with the Modal training runtime's forced CPU auth-eval
```

Evidence basis:

- The recipe advertises `target_modes: contest_exact_eval`,
  `predicted_band_axis: contest-CUDA`, and an exact path ending in
  `contest_auth_eval.py --device cuda`.
- `experiments/modal_train_lane.py` forces worker env
  `AUTH_EVAL_DEVICE=cpu`, `MODAL_AUTH_EVAL_ADVISORY_ONLY=1`,
  `SCORE_CLAIM=false`, and `PROMOTION_ELIGIBLE=false`.
- `experiments/train_substrate_time_traveler_l5_z6.py` calls the canonical
  smoke auth-eval gate as a legacy CUDA-claim caller. Under explicit
  `AUTH_EVAL_DEVICE=cpu`, the gate runs CPU advisory auth-eval and returns
  `None`, so the smoke-before-full wrapper's `contest_cuda_auth_eval_v1`
  validator should never receive a valid contest-CUDA smoke claim from this
  Modal path.

Interpretation: Candidate 4c is still the top queue candidate and the
Catalog #202 env-attested launch surface is mechanically coherent, but the
current Modal smoke/full route is not contest-CUDA compatible. Do not launch
this recipe as a paid contest-CUDA Modal smoke until one of these is true:

1. route post-training exact CUDA eval through a provider/runtime path that
   does not force `AUTH_EVAL_DEVICE=cpu`; or
2. explicitly downgrade this recipe to a diagnostic/training-artifact smoke
   contract with `score_claim=false`, then create a separate claimed exact-CUDA
   eval handoff for the emitted byte-closed archive/runtime packet.

Signal-loss repair:

- `tools/run_codex_review_for_dispatch.py` now parses bracketed lowercase
  severity markers like `- [high] ...`, not only `HIGH:` style lines.
- `tools/verify_candidate4c_launch_packet.py` now consumes the latest
  Candidate 4c codex-review artifact and refuses paid-execution readiness when
  the verdict is `needs-attention`, `no-ship`, or `invocation-error`.
- The packet doctor recovers findings from `raw_output_excerpt` when older
  review JSON has an empty `findings` array.

Verification:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_verify_candidate4c_launch_packet.py \
  src/tac/tests/test_check_271_pre_dispatch_codex_review.py::TestVerdictParsing
# 28 passed in 0.54s

.venv/bin/python -m pytest -q \
  src/tac/tests/test_verify_candidate4c_launch_packet.py \
  src/tac/tests/test_check_271_pre_dispatch_codex_review.py \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py
# 125 passed in 16.79s

.venv/bin/python -m py_compile \
  tools/verify_candidate4c_launch_packet.py \
  src/tac/tests/test_verify_candidate4c_launch_packet.py \
  tools/run_codex_review_for_dispatch.py \
  src/tac/tests/test_check_271_pre_dispatch_codex_review.py \
  tools/asymptotic_pursuit_dispatch_queue.py
# PASS

git diff --check
# PASS

.venv/bin/python tools/check_dispatch_cli_shell_hazards.py --strict
# PASS

OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1 \
OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=catalog202_sentinel_audit:a04c09b40cca80dd98c41968770ce7d2b19b672e2553d6a2bbb5c03b1a5aa387 \
OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON=.omx/state/catalog202_sentinel_cleanliness/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch_20260518T072226Z.json \
OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=13.000 \
.venv/bin/python tools/verify_candidate4c_launch_packet.py --json --write-artifact \
  --queue-path .omx/state/asymptotic_pursuit/dispatch_queue_20260518T091517Z.json
# rc=1; wrote candidate4c_no_spend_launch_packet_20260518T092725Z.json
# ready_for_operator_paid_execution=false
# queue_immediate_launch_ready=true
# codex_pre_dispatch_review_ready=false
# codex_pre_dispatch_review_blockers=[
#   candidate4c_codex_pre_dispatch_review_blocking_needs-attention
# ]
```

No provider job was launched and no lane claim was opened.

## 2026-05-18 Codex Asymptotic Queue Cost-Model Repair

Adversarial queue review found a paid-launch accounting bug in
`tools/asymptotic_pursuit_dispatch_queue.py`: the readiness estimate
`estimated_dispatch_cost_usd` already includes the paired CPU axis from
`_estimate_dispatch_cost(..., paired_axis=True)`, but the dispatch sequence also
added an explicit `paired_cpu_axis_verification` stage. Candidate 4c therefore
showed a ready rollup of `$2.183` instead of the additive stage total implied by
smoke + contest-CUDA full run + one paired CPU axis.

The dispatch sequence now splits the paired estimate into:

- `smoke_100ep`: `$1.000`
- `full_eval_contest_cuda`: `$0.983`
- `paired_cpu_axis_verification`: `$0.100`

Candidate 4c remains TOP-1 READY, but the ready paid-launch rollup is now
`$2.083`, not `$2.183`. This is cost/custody hardening only, not score evidence.

Verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py -q
# 44 passed in 1.09s

.venv/bin/python tools/asymptotic_pursuit_dispatch_queue.py --json \
  | jq '{top_1_substrate, ready_total_cost_usd_if_dispatched:.cost_band_rollup.ready_total_cost_usd_if_dispatched, candidate4c:([.dispatch_sequence[] | select(.substrate_id=="z6_v2_candidate_4c_scorer_logit") | {total_estimated_cost_usd, stages}] | .[0])}'
# top_1_substrate=z6_v2_candidate_4c_scorer_logit
# ready_total_cost_usd_if_dispatched=2.083
# candidate4c.total_estimated_cost_usd=2.083
```

No provider job was launched and no lane claim was opened.
