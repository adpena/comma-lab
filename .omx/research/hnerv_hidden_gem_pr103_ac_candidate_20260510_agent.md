# HNeRV Hidden-Gem PR103-ac Candidate - Local Materialization

generated_at: 2026-05-10
research_only: true
score_claim: false
dispatch_attempted: false
lane_claim_created: false

## Summary

Materialized a local-only byte-different candidate from the refreshed HNeRV
scorecard route:

- scorecard inputs:
  `experiments/results/hnerv_frontier_scorecard_refresh_20260510_codex/scorecard.json`
  and `section_profiles.json`
- routed target: `PR103-ac-repack` /
  `merged_range_coded_weights_and_hi_latents`
- output directory:
  `experiments/results/hnerv_hidden_gem_pr103_ac_candidate_20260510_agent/`
- candidate archive:
  `experiments/results/hnerv_hidden_gem_pr103_ac_candidate_20260510_agent/archive.zip`
- manifest/proof:
  `manifest.json`, `runtime_closure.json`, `runtime_decode_proof.json`

The candidate removes one 4-byte word at payload offset `160824`, inside both
the scorecard target slice and the runtime-consumed PR103-on-PR106 `merged_ac`
range stream.

## Byte And Section Proof

- source archive: `185578` bytes,
  SHA-256 `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- candidate archive: `185574` bytes,
  SHA-256 `8274e88c0ab1d26a06470a0730d17fe004556afa564460cf1c05624ff6060278`
- charged archive delta: `-4` bytes
- payload delta: `-4` bytes
- ZIP overhead delta: `0` bytes
- scorecard profile section old SHA-256:
  `11e9614692d36a07db05787f6153d1640f480d1fd896eb4019628adee3f8371b`
- scorecard profile section new SHA-256:
  `d567b618e7a83cf2651fef12deac7011632aab1f85a5449fe296a337ed0d8ed4`
- runtime `merged_ac` old SHA-256:
  `1ad4e883ac8fef97af7f07d24aff790e5fd2d52cb05f5a4ee3a58e4e4b7b315c`
- runtime `merged_ac` new SHA-256:
  `5c2cdbc730f7a801c144c46a74f51c99fda972c5eb61c4029ccb32fe232e3f06`

The apparent scorecard section is a fixed profile slice, while the exact
runtime consumes the PR103-on-PR106 `0xff | decoder_len | decoder | latents`
packet. The manifest records both names so future review does not confuse the
profile label with the runtime closure.

## Runtime Consumption / No-Op Proof

`runtime_decode_proof.json` passed locally:

- candidate parses with `tac.pr103_pr106_runtime_closure`
- generated local runtime adapter parses the same candidate bytes
- generated runtime decode matches TAC closure decode
- latents are exact versus source
- decoded state dict changes in three tensors:
  `blocks.4.weight`, `blocks.5.weight`, `refine.0.bias`
- max state-dict absolute diff versus source: `0.82373046875`
- runtime static scan found no scorer/TAC imports
- generated runtime `inflate.sh` passes `bash -n`
- generated runtime dependency check passes with brotli/constriction/numpy/torch

This is therefore not a cosmetic ZIP repack or provenance-only edit. The
changed bytes are inside the consumed range-coded stream and alter decoded
runtime tensors. Distortion is unknown until exact CUDA auth eval.

## Solver Wire-In Declarations

- Sensitivity-map contribution: N/A for this landing; no scored empirical
  anchor was produced.
- Pareto constraint: non-binding; no score claim and decoded state changed.
- Bit-allocator hook: N/A; this is a byte-level range-stream deletion probe,
  not a learned allocation update.
- Cathedral autopilot dispatch hook: intentionally not registered; no remote
  dispatch per operator scope. Future exact eval must claim a lane first.
- Continual-learning posterior update: not updated; no scored empirical anchor.
- Probe-disambiguator: not required. The only ambiguity found was profile
  section name versus runtime-consumed section, captured explicitly in the
  manifest.

## Remaining Blockers Before Any Score Claim

1. Claim the relevant lane before any exact CUDA dispatch.
2. Run the generated packet through exact CUDA auth eval.
3. Treat the candidate as score-unknown because decoded state tensors changed.
4. Do not promote from the `-4` byte rate term alone.
