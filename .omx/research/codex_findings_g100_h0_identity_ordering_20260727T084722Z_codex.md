# Codex findings — G100 H0 identity ordering

UTC: `2026-07-27T08:47:22Z`  
Lane: `lane_g100_lattice_teacher_compaction_takeoff_20260727`  
Verdict scope: G100 H0 and the unlaunched V2 same-solution recode only

## Authority verdict

H0 is a valid, dense-free custody/index result. It proves the exact historical
MS2R packet and MS1 evidence population were joined under their immutable
hashes, with no teacher pixels persisted or made payload-eligible. It is not a
candidate, a score result, or a proof that V2 decodes the same population.

## Findings and fixes

1. **Index root was at risk of being used as an output root.** The H0
   `content_root_sha256=073d079e...` hashes index leaves and source-custody
   identities. It does not hash decoded frames. The historical packet's
   `decoded_two_plane_sha256=3494c0cf...` is plane-major, while V2's decoded
   root is pair-major. Cross-order root comparison would be false authority.
   The specification now requires a streaming source pair-major root plus
   per-frame identity hashes before any H1 equality claim.
2. **Double replay was not universal.** The whole-object ledger rejected
   differing replay roots for `RECODE_IDENTITY`, but baseline and lossy rows
   could enter despite nondeterministic receiver output. The ledger now
   requires `deterministic_replay_sha256 == output_root_sha256` for every
   action kind before scoring or inheritance.
3. **Forged component lengths could request an unbounded read.** The V2 decoder
   now proves each pair's two declared bodies fit inside the already capped
   physical payload before reading them.

Regression coverage exercises nondeterministic baseline refusal and oversized
pair-body refusal. The focused H0/V2/homotopy suite passes 7/7; Ruff, format,
and Python compilation are clean.

## Continuation decision

H1 remains frozen. Re-arm only after a dense-free materializer:

- streams the original 600 selected pairs without retaining the population;
- derives source pair-major and per-frame hashes in that same pass;
- builds the V2 payload twice with identical bytes;
- strict-decodes twice and matches every pair/slot plus the pair-major root;
- wraps the public receiver object and prices actual final archive bytes.

Pointer delta: none.

STORES CONSULTED: G21 specification; real H0 config and receipt; SSD immutable
stage receipts; H0 index, V2 codec, whole-object ledger, governed runner, and
focused tests.

HISTORICAL_PROVENANCE: first Codex adversarial review of the real G100 H0
takeoff; preserves the valid custody result while extinguishing decoded-root
ordering confusion and non-universal replay proof.
