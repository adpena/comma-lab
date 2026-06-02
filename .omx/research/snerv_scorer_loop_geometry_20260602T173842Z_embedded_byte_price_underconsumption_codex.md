# SNeRV scorer-loop geometry

Schema: `snerv_scorer_loop_geometry.v1`
Authority: `false_authority_macos_cpu_snerv_scorer_loop_geometry_no_score_claim`
Inputs: `5`
Best descent delta: `-0.02724836167734246`
Best descent input: `/Volumes/VertigoDataTier/pact/snerv_scorer_loop_qat_local_20260602T170918Z_score_primary_4pair8trial/snerv_scorer_loop_qat_local_result.json`
Lowest local score: `0.5499757754168919`
Lowest local score input: `/Volumes/VertigoDataTier/pact/snerv_scorer_loop_qat_local_20260602T170501Z_score_primary/snerv_scorer_loop_qat_local_result.json`

## Aggregate

- Best search mode: `learned_random_subspace`
- Dominant lowering axis: `pose`
- Accepted trial count: `8`
- Evaluated trial count: `37`
- Rate is current descent driver: `False`

## Inputs

### `snerv_scorer_loop_qat_local_20260602T165640Z`

- Search: `learned_random_subspace`
- Pairs: `1`
- Score: `0.5505918809022246` -> `0.5505918809022246`
- Delta: `0.0`
- Seg contribution delta: `0.0`
- Pose contribution delta: `0.0`
- Rate contribution delta: `0.0`
- Geometry verdicts: `[]`

### `snerv_scorer_loop_qat_local_20260602T170501Z_score_primary`

- Search: `learned_random_subspace`
- Pairs: `1`
- Score: `0.5505918809022246` -> `0.5499757754168919`
- Delta: `-0.0006161054853326409`
- Seg contribution delta: `0.0005086185410618782`
- Pose contribution delta: `-0.0011193971547695802`
- Rate contribution delta: `-5.326871624994345e-06`
- Geometry verdicts: `['score_primary_found_local_descent', 'pose_geometry_primary_current_descent', 'component_tradeoff_admitted_by_lagrangian', 'receiver_replayed_accepted_candidate_exists', 'rate_not_current_descent_driver']`

### `snerv_scorer_loop_qat_local_20260602T170543Z_score_primary_4pair4trial`

- Search: `learned_random_subspace`
- Pairs: `4`
- Score: `1.1927130470929876` -> `1.1787591662790842`
- Delta: `-0.013953880813903474`
- Seg contribution delta: `-0.0012715638149529696`
- Pose contribution delta: `-0.012684980434763002`
- Rate contribution delta: `2.663435812455539e-06`
- Geometry verdicts: `['score_primary_found_local_descent', 'pose_geometry_primary_current_descent', 'receiver_replayed_accepted_candidate_exists', 'rate_not_current_descent_driver']`

### `snerv_scorer_loop_qat_local_20260602T170918Z_score_primary_4pair_nes4`

- Search: `nes_pair_robust`
- Pairs: `4`
- Score: `1.1927130470929876` -> `1.1858877584913716`
- Delta: `-0.00682528860161602`
- Seg contribution delta: `-0.0011444091796875`
- Pose contribution delta: `-0.005678215986116314`
- Rate contribution delta: `-2.663435812455539e-06`
- Geometry verdicts: `['score_primary_found_local_descent', 'pose_geometry_primary_current_descent', 'receiver_replayed_accepted_candidate_exists', 'rate_not_current_descent_driver']`

### `snerv_scorer_loop_qat_local_20260602T170918Z_score_primary_4pair8trial`

- Search: `learned_random_subspace`
- Pairs: `4`
- Score: `1.1927130470929876` -> `1.1654646854156452`
- Delta: `-0.02724836167734246`
- Seg contribution delta: `-0.0002543092705309391`
- Pose contribution delta: `-0.02699405240681152`
- Rate contribution delta: `0.0`
- Geometry verdicts: `['score_primary_found_local_descent', 'pose_geometry_primary_current_descent', 'receiver_replayed_accepted_candidate_exists', 'rate_not_current_descent_driver']`

## Blockers

- `snerv_scorer_loop_geometry_is_false_authority`
- `paired_contest_cpu_cuda_auth_eval_missing`
- `full600_receiver_proof_required`
