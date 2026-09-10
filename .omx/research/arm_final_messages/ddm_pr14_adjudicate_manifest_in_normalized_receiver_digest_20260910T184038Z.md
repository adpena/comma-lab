Completed the adjudication memo: [ddm_pr14 memo](/Users/adpena/Projects/pact/.omx/research/ddm_pr14_adjudicate_manifest_in_normalized_receiver_digest_20260910.md). It is 48,452 bytes, SHA-256 `6e1732baea1912731fef6a53abca0cb94faf2e9440accc304dd9b34e593bade9`.

1. `MANIFEST.sha256` receiver content — **AMEND (a):** exclude it only from the new versioned pre-fire timing-risk digest.
2. Reference re-derivation — **AMEND:** RLC1 and RLC4 both recompute to `9f6e71680a13d8598974ee13f78a1a72759a758681e6d86b7b288cc105442890`.
3. Evidence reuse — **AMEND:** reuse immutable production receipts; regenerate the risk delta, timing-risk receipt, and never-created intent.
4. Move 40 legacy leg — **RATIFY-AS-IS:** `tac.decode_wall_clock`, receiver `6726fd77…`, and `t4_direct` remain untouched.
5. Freeze update — **AMEND:** append a measured amendment row after the implementation landing; never rewrite the existing freeze.

| Endpoint | Legacy digest | Amended digest | Rows | Raw/normalized bytes |
|---|---|---|---:|---:|
| RLC1 reference | `b06e59a…1e2d` | `9f6e7168…2890` | 49 | 824,789 / 824,743 |
| RLC4 candidate | `27948d3d…bb8a` | `9f6e7168…2890` | 49 | 824,789 / 824,743 |

The 49 normalized rows are byte-identical. The memo records every input pin, both raw pinset digests, both excluded manifest hashes, the receipt-by-receipt reuse table, literal implementation contract, six named tests, freeze schema, and the non-pin `inflate.py` falsifier.

No source, runtime, volume, pointer, or staged-index content was changed. The existing focused tests passed: `115 passed`. No amended tests, timing run, scorer, evaluator, authorization, or fire occurred.

Serializer: one attempt, `rc 17` from Git object-write denial. MAIN-ready fallback commit `8dd2267e7fef40a672bdb54de104147a53153e00` is in [intended-commit.bundle](/Volumes/VertigoDataTier/pact/anonymous/receipts/commit_serializer_fallbacks/20260910T183933.751566Z-67498/intended-commit.bundle), SHA-256 `cfb38870a3463025bf134f99b124494e711029f4a61829edaf6de314eb00a9d0`. The shared index remained unchanged.

Frontier unchanged: `composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: shared repository; fire trigger: verify and land fallback commit `8dd2267e…`.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN / FFI implementation arm; consumer store: amended `candidate_seal.py`, tests, and implementation manifest; fire trigger: the memo is committed.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN / FFI implementation arm; consumer store: `PREFIRE_CONTRACT_FROZEN.json`; fire trigger: implementation tests pass and its commit/manifest/memo hashes are measured.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: rlc5 producer; consumer store: new normalized-risk delta, timing-risk receipt, and intent; fire trigger: amendment and freeze commits exist, pointer remains move 42, and reused receipts revalidate.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: committed authorization and retained T4 output; fire trigger: intent validates, refusal controls pass, cost is below $5, lane is claimed, and cloud state is clear.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `candidate_seal.v3`, evaluation ledger, and pointer packet; fire trigger: harvested exact result passes unchanged `t4_direct`, custody, and score-bar checks.

## LIVE-HYPOTHESES

- The 180,178-byte candidate may beat move 42 because twin encodes and the full cold public output agree; exact T4 scoring remains untested.
- It may finish below 1,260 seconds because retained diagnostics estimate 1,032.725 seconds; that estimate is not timing authority.
- The scoped digest should remain cross-actor stable because both retained roots already produce the same 49 normalized rows; the real intent/authorization path remains untested.

## DEAD-ENDS

- Verbatim manifest inclusion in the risk digest is closed: it reintroduces normalized archive pins through a derived hash.
- Restoring RLC1’s stale manifest is closed: it would violate PR9’s raw-manifest requirement.
- Changing or rebuilding the legacy move-40 timing leg is closed: the versioned risk digest avoids touching it.
- Excluding the manifest from full runtime identity or custody checks is closed: those checks make the scoped exception safe.
- Re-running immutable production evidence merely to obtain new timestamps is closed: exact byte/hash reuse is the valid path.
- Treating conditional score, diagnostic timing, fixtures, or intent emission as promotion evidence is closed: only a harvested exact row can move the pointer.