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
- Fix: Stage 4 records `REMOTE_DRIVER_STAGE4_STARTED_UNIX` immediately before
  invoking the trainer. Stage 5 now rejects `stats.json` with mtime older than
  that timestamp and exits `33` before writing `LANE_Z6_PCWM_DONE`.
- Dispatch-ledger effect: stale stats terminalize as
  `failed_z6_pcwm_remote_driver_rc_33` with
  `evidence_marker=[not-yet-classified] score_claim=unknown`, preserving the
  fact that the current invocation did not produce usable evidence.

This blocks a stale-output false authority path without changing model training
or archive bytes.

## Verification

- `bash -n scripts/remote_lane_substrate_time_traveler_l5_z6.sh`
  - PASS after remote completion missing-stats fail-closed repair
- `.venv/bin/python -m pytest src/tac/tests/test_time_traveler_l5_z6_remote_driver.py -q`
  - PASS after remote completion missing-stats fail-closed repair: `3 passed in 0.34s`
  - PASS after remote terminal-claim axis split repair: `3 passed in 0.32s`
  - PASS after remote completion stale-stats reuse repair:
    `4 passed in 0.41s`
- `.venv/bin/python -m pytest src/tac/tests/test_z6_v2_candidate_1_wave_2_build.py src/tac/tests/test_time_traveler_l5_z6_remote_driver.py src/tac/tests/test_probe_z6_predictive_coding_vs_identity_disambiguator.py -q`
  - PASS after remote completion missing-stats fail-closed repair:
    `50 passed in 13.22s`
  - PASS after remote terminal-claim axis split repair:
    `50 passed in 11.88s`
  - PASS after remote completion stale-stats reuse repair:
    `51 passed in 12.55s`
- `.venv/bin/python tools/canonical_dispatch_optimization_protocol.py --trainer experiments/train_substrate_time_traveler_l5_z6.py --recipe substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch --json`
  - PASS after remote completion missing-stats fail-closed repair:
    `overall_pass=true`, `blockers=[]`, Tier 1/2/3 blockers all `[]`
  - PASS after remote terminal-claim axis split repair:
    `overall_pass=true`, `blockers=[]`, Tier 1/2/3 blockers all `[]`
  - PASS after remote completion stale-stats reuse repair:
    `overall_pass=true`, `blockers=[]`, Tier 1/2/3 blockers all `[]`
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
