# Codex Findings - PSV4 Selector Adapter

Generated: 2026-05-28T12:24:00Z

## Verdict

PSV4 PACT-NeRV selector packets are now an implemented score-affecting repair
adapter class. The executor decodes the PSV4 RLE selector stream, mutates
family-targeted pair selectors, re-encodes the selector, rewrites the packet
header length, repacks the archive, and writes an archive-bound receiver parse
proof.

Authority remains fail-closed: generated rows are MLX/local advisory, exact
handoff remains blocked on contest CPU/CUDA custody, and proxy rows cannot claim
score, promotion, rank/kill, dispatch, or budget authority.

## Live Proof

- Source archive:
  `experiments/results/pact_nerv_selector_v4_mlx_full_layoutfix_20260528Tlocal/archive.zip`
- Source SHA-256:
  `a8ebb822e50a41b1d00ef512ebf8a0ea006bac2323f57e3e94088d3594bd8171`
- Proof artifact:
  `.omx/research/psv4_selector_adapter_live_proof_20260528T122327Z/summary.json`
- Selected transform:
  `psv4_selector_payload_mutation`
- Candidate emitted:
  `.omx/research/psv4_selector_adapter_live_proof_20260528T122327Z/byte_transform/candidate_archive_psv4_selector_payload_mutation.zip`
- Runtime proof:
  `.omx/research/psv4_selector_adapter_live_proof_20260528T122327Z/byte_transform/candidate_archive_psv4_selector_payload_mutation_receiver_proof.json`

The live MLX archive had a 32-pair PSV4 selector stream with all selectors at
code 0. The SegNet waterfill mutation changed 16 pair selectors to code 2,
increasing selector bytes from 2 to 64 and archive bytes by 27. That is useful
negative/interaction signal: semantic mutation is now executable, while byte
credit exhaustion and stack penalty must decide whether this family is worth
promotion in a given chain.

## Engineering Closure

- `pact_nerv_selector_v4_packet` moved from unsupported next-adapter queue to
  implemented score-affecting archive adapter registry.
- Automation rows now route PSV4 to `psv4_selector_payload_mutation` with byte,
  selector, pair, frame, batch, and full-video scopes.
- The executor ranks semantic packet mutations above non-semantic ZIP repacks
  while preserving false-authority overlays.
- Focused tests cover registry rollup, synthetic PSV4 mutation, existing FP11
  selector adapters, and FEC8 Markov selector mutation.

## Remaining High-EV Gaps

The next score-affecting unsupported families are PSV3, HDM latent sidecars,
renderer DFL1/RPK1/ASYM payloads, raw HNeRV payloads, STC raw payloads, and
tensor-factorized payloads. They should follow the same pattern: packet parser,
semantic mutator, archive-native repack, receiver proof, exact-axis handoff
blocker, and posterior ingestion of byte/score interactions.
