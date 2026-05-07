# PR91/HPM1 Categorical Adversarial Review - 2026-05-07

Scope: PR91/HPM1, QMA9, CLADE/SPADE/openpilot categorical mask priors, and
whether this lane can realistically contribute to a `0.18` / `0.15` frontier
push. This is forensic only: `score_claim=false`, `dispatch_allowed=false`.

## Evidence Base

- PR91 archive custody is clean: `archive.zip` is `222404` bytes with SHA-256
  `4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f`.
- HPM1 mask payload custody is clean: `tokens_len=116796`,
  `tokens_sha256=541016d83852a5bb3e0738caa3b44d7b2b0f7372f1841085cf9554f039c6cf6b`,
  `hpac_len=28243`,
  `hpac_sha256=de7638c531c9dafa06148602cf784bf3ae9997f326f85cc25b9f3646b536abdd`.
- HPAC model load works locally: PPMd state decompresses and `HPACMini` loads
  on CPU, but source-order prefix decode fails at frame `0`, group `10`,
  symbol `191`, decoded-before `5951`.
- Spatial/order probes:
  - `source_mask_row_major`: fails at decoded-before `5951`.
  - `full_col_major`: fails earlier at decoded-before `2201`.
  - `phase_major_row_major`: advances to decoded-before `6926`.
  - `tile_major_row_major`: advances to decoded-before `8274`.
- PR85/QMA9 teacher forcing is not a clean bridge:
  - tile-major reference forcing regresses to decoded-before `2033`.
  - phase-major reference forcing reaches decoded-before `15989`, but still
    fails, with `5595` decoded/reference mismatches before failure.
- Same-group suffix scan at the tile-major failure tested `1134` remaining rows
  and found `0` decodable rows.
- New failure-row probability scan tested `192` categorical variants
  (`float32/float64`, `perfect` true/false, `prob_eps` from `1e-12` to `1e-2`,
  uniform mix from `0` to `1e-2`) against the cloned failure state and found
  `0` decodable rows.

Primary artifacts:

- `experiments/results/pr91_hpm1_readiness_20260506_codex/readiness.json`
- `experiments/results/pr91_hpm1_runtime_contract_20260506_codex/runtime_contract.json`
- `experiments/results/categorical_openpilot_payload_candidate_20260506_codex/readiness.json`
- `.omx/research/pr91_hpm1_submitted_prefix_token_recovery_tile_major_20260507_codex.json`
- `.omx/research/pr91_hpm1_next_row_suffix_scan_tile_major_20260507_codex.json`
- `.omx/research/pr91_hpm1_failure_row_probability_scan_tile_major_20260507_codex.json`

## Bug And No-Op Review

### HPM1 semantic parity

The lane has byte custody and structural payload parsing, not semantic closure.
`full_decode_600_frames=false` and `byte_exact_reencode=false` remain the
central blockers. A structural HPM1 reencode proves only the header/tokens/HPAC
slice grammar; it does not prove class masks, frame output, or range reemit.

No-op risk: treating `categorical_payload.bin` as "consumed" when the runtime
only parses and hashes charged members. That is useful custody, but not a
decoder and not a replacement. Any candidate that does not produce decoded
masks and re-emit exact HPM1 bytes is non-promotable.

### Range/probability contract

The new failure-row scan narrows a simple numeric categorical mismatch. At the
tile-major failure row, the normalized row SHA-256 is
`8216c3d82263ef0fc10c88ddf28439b0916ae83865c8d14d9e37bd785bd2b7cd`, argmax
class `2`, but no tested `prob_eps` / float / `perfect` / uniform-mix variant
decodes from the cloned state. That moves the likely bug class earlier than the
failed row: prior context/order drift, range construction/finalization drift,
or true encoder symbols not represented by public runtime semantics.

No-op risk: continuing to tune the failure row after this scan. The current
evidence says row-local smoothing is not a material unlock.

### QMA9 reference semantics

PR85/QMA9 tokens are a useful reference tensor, not proven PR91 encoder tokens.
The first decoded/reference mismatch appears at global symbol `7`, and
tile-major teacher forcing regresses. Phase-major teacher forcing is the only
longer prefix, but it still fails at decoded-before `15989` and accumulates
thousands of mismatches.

No-op risk: claiming a PR85/QMA9 bridge from visual or local-reference
agreement. It must pass full HPM1 decode/reencode parity before it can support
archive replacement.

### CLADE/SPADE/openpilot labels

The label contract itself is in decent shape: contest classes are zero-based
comma10k order (`road`, `lane_markings`, `undrivable`, `movable`, `my_car`),
and the Selfcomp grayscale codebook is treated as a wire codebook, not a
semantic relabeling. The categorical readiness artifact also distinguishes
charged runtime conditioning from compression-time-only openpilot priors.

The blocker is not label naming. The blocker is runtime value: CLADE/SPADE and
openpilot priors need charged archive consumption and output/parity proof.
Current readiness says `decode_reencode_parity_not_passed`,
`decode_reencode_full_decode_not_proven`, and
`decode_reencode_byte_exact_reencode_not_proven`.

No-op risk: shipping class codebooks, label manifests, or openpilot atom rankers
that never affect inflate output. Compression-time-only rankers can guide a
separate candidate, but they cannot justify exact eval unless a charged runtime
consumer changes decoded masks and passes no-op controls.

### Runtime closure

PR91 runtime source inventory exists, but the HPAC device contract is
contradictory: source comments say CPU is required for bit-exact arithmetic,
while the visible call site passes ambient `device`. The runtime audit still
blocks on `hpac_device_contract_resolved` and
`runtime_consumer_sidecar_free_hpm1`.

No-op/parity risk: exact archive bytes can score differently if the local
runtime tree changes. A promotable path needs archive-only inflate, no sidecars,
deterministic CPU/CUDA device contract, runtime tree SHA, and exact CUDA auth
eval only after local full parity.

## 0.18 / 0.15 Realism

The idea is high-upside in theory because a small categorical mask grammar could
replace a major charged stream. In shortest wall-clock practice, HPM1 is not a
credible contributor to `0.18` or `0.15` from the current evidence. The lane is
still missing the exact thing that would make it a frontier replacement:
semantic full decode plus byte-exact reencode and a sidecar-free runtime.

The best measured local prefix, tile-major, reaches only `8274` symbols before
range failure. The most favorable teacher-forced phase-major diagnostic reaches
`15989` symbols but is off-contract and mismatch-heavy. The new failure-row scan
did not find a bounded numeric repair. That means further local blind probing is
likely to spend wall-clock without producing an exact-evaluable candidate.

## Continue Criteria

Continue HPM1 only if at least one of these appears:

- Real PR91 encoder source, encoder trace, or saved probability/symbol rows
  that can be checked against the submitted `uint32` token stream.
- A bounded probe that replays materially past the current best prefix under a
  source-plausible contract and produces a deterministic path toward full
  decode/reencode.
- A byte-exact range reemit proof for a nontrivial prefix against submitted
  words, not just structural header equality.
- A charged runtime consumer that actually emits masks from archive members,
  passes label permutation/identity no-op controls, and records sidecar-free
  runtime closure.

## Stop / Redirect Criteria

Stop HPM1 wall-clock now unless the continue criteria are met. Recommended
redirect: prioritize other frontier replacement or categorical candidates with
clearer exact-evaluable paths, and keep HPM1 as a forensic source of constraints
for future categorical-mask work.

Do not dispatch HPM1 exact eval. Do not claim score. Do not treat PR91/QMA9 or
CLADE/SPADE/openpilot labels as promotable until full decode/reencode parity and
runtime closure exist.
