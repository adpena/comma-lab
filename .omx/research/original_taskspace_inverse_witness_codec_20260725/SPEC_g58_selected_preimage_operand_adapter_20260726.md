# G58 selected-preimage program operand adapter

Date: 2026-07-26  
Lane: `lane_g58_selected_preimage_program_operand_adapter_20260726`  
Authority: structural implementation only; no score, rate, or candidate claim

## Objective

Land the smallest production bridge from
`TaskspaceSelectedPreimageProgramV1` plus a
`TaskspaceSelectedPreimageDecoderV1` into the `FreshOperandProviderV1`
protocol consumed by `taskspace_fresh_selected_plane_codec_v1`.

The selected-preimage program is the sole producer of scorer-resolution
`Y0/Y1`. A separately bound fresh operand provider supplies only chronological
pair IDs, exact batch-16 target labels, and source-cache poses that remain
advisory. Its direct `Y0/Y1` arrays are ignored and must never leak into the
adapter output. Value equality is not leakage: a selected representative may
legitimately equal the direct source plane. Custody is proven by G49 decode
construction and disjoint storage, not an invalid whole-array inequality test.

## Production contract

- Production is exactly n600 in five chronological 120-pair stages.
- Production recursively reopens the exact G51 aggregate schema, self-seal,
  teacher batch-16 binding, five stage manifests and their Y0/Y1/pose file
  identities. Caller-supplied target/pose digests are not authority.
- Each stage is decoded through `iter_selected_preimage_segment`; there is no
  dense n600 bank and no hidden historical fallback.
- The exact selected-preimage packet bytes/SHA, fresh V15 semantic compile
  identity/bytes, batch-16 target custody identity, generic V10 decoder source
  identity, and advisory label/pose input identities are bound.
- At least one counted factor must address a pair and change decoded behavior.
  G49 remains the decode authority for per-addressed-factor non-inertness.
- Labels, poses, target banks, scorer weights, and historical payloads are
  forbidden counted outputs. The adapter carries only their custody hashes.
- Factor payload bytes are explicitly reported as byte homes *inside* the
  selected-preimage packet. The truthful pre-container counted source total is
  `semantic_archive_bytes + program_packet_bytes`; factor bytes must not be
  omitted or added a second time.
- Small fixture populations are allowed only behind an explicit test-only
  switch and have no empirical authority.

## `PROGRAM_RESIDUAL_LAYERED` fail-closed boundary

The provider bridge alone is not `PROGRAM_RESIDUAL_LAYERED`. That name is
available only after a typed outer-archive proof reopens a real regular ZIP and
verifies:

1. one member equals the exact fresh semantic archive bytes/SHA;
2. one distinct member equals the exact selected-preimage packet bytes/SHA;
3. parse-back reproduces the exact program and all factor byte homes partition
   the packet;
4. every learned factor binds the callable decoder contract and implementation
   source required by its counted payload (analytic-only programs satisfy this
   condition vacuously and remain legal);
5. the counted outer ZIP has exact bytes/SHA and its complete member set is
   exactly the two typed counted members above. Any third, duplicate, encrypted,
   directory, symlink-like, or otherwise untyped member fails closed regardless
   of its name. This complete partition prevents neutral-name laundering of
   targets, poses, scorer state, direct planes, or historical payload.

Without that proof the adapter reports
`PROGRAM_RESIDUAL_LAYERED_BLOCKED_OUTER_EMBEDDING_OWED`.

The production lifecycle is three separate durable artifacts: the pre-encode
identity, a post-iteration terminal stage-chain receipt, and the physical outer
proof. The terminal receipt is unavailable until the whole lattice has
completed; it binds the campaign receipt, exact published identity file, G51
aggregate file/self-seal, every complete admission row, terminal chain, and its
own self-seal. No pre-iteration artifact may claim a terminal chain.

G59 consumes these artifacts only through
`reopen_program_residual_production_pre_encode_evidence(identity_path=...,
terminal_stage_chain_path=..., outer_proof_path=...)`. The strict verifier
reopens each identity and payload from one no-follow descriptor, checks the
exact n600/120x5/batch16 chain, parses the counted G49 packet, verifies all byte
homes, binds the fresh semantic/compiler/target/generic-V10 sources, and
reopens every learned decoder source against the contract and source SHA
encoded in that factor. An analytic-only packet is legal with empty learned
source/contract sets. It independently re-enumerates the exact two-member outer
archive from the same descriptor-stable bytes rather than trusting the proof's
absence booleans. Published receipts are true write-once hard-link landings;
symlinks, replacement, and raced differing writers fail closed.

## Verification

Focused fixtures must prove:

- chronological stage composition and preservation of the five-stage
  production lattice;
- G49 decoded planes, rather than advisory-provider planes, reach G52;
- value-equal selected representatives are admitted while direct-plane storage
  aliasing is refused;
- packet, semantic, target-custody, decoder, label, and pose hashes bind;
- a missing/inert counted factor or identity mismatch fails closed;
- an archive missing or mutating either counted member cannot obtain the
  program-residual representation claim;
- a physically embedded, source-bound fixture reopens deterministically.
- analytic-only factors reopen without inventing a learned decoder obligation;
- mutating any admission/custody field breaks the recurrent chain.

No heavy encode, scorer, training, or full-n600 job is part of G58.
