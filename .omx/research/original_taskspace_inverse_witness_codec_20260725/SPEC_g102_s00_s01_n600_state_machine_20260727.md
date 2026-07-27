# G102 S00 -> S01 full-n600 state machine (2026-07-27)

Status: `BUILD + local-verify ONLY`; `research_only=true`; no heavy launch; no
score/frontier/pointer mutation.

## Purpose

`experiments/run_g102_semantic_root_s00_s01_n600_v1.py` is the physical
orchestrator for the first G102 `SemanticRootY1V1` row. It deliberately does
not implement the missing packet/compiler/receiver. It records S00 custody and
refuses S01 until a fresh, own-lineage, P-free compiler plus its public codec
section exist under exact source hashes.

The runner does not own or reproduce compiler placement, population mapping,
obligation ownership, proof dependency, or lifecycle semantics. Those remain
single-sourced in
`src/tac/witness_dsl/taskspace_selected_solution_compiler.py`. Every S01 stage
must return the exact canonical `G17WholeObjectStateV1`. The runner then uses
G17's canonical builder and strict parser and requires the receipt to bind the
same complete archive bytes, source/global pair IDs `0..599`, n600 score
observation, component distortions, archive byte count, coupled score, and the
exact packet-bound source-lineage manifest bytes as encoder-only placement
evidence. Anything else refuses.

## State contract

### S00_CUSTODY

- Load and verify the dynamic competitive-target pointer; never hardcode a
  frontier.
- Verify exact `upstream/evaluate.py` source custody.
- Strict-reopen G46's canonical batch-16 audit and compile-ready primary
  materialization receipt, including all 600 checkpoints, source video,
  SegNet weights, target-label bank, exact file hashes, and both receipt seals.
- Select storage by the governed waterfall: VertigoDataTier, then APDataStore;
  local disk remains disabled by default.
- Reserve at least two full 1200-frame raw outputs for deterministic
  double-decode.
- Seal seed, typed config, explicit compiler/runtime/G46 paths and hashes, the
  derived source-lineage policy, upstream snapshot identity, storage plan,
  exact five-by-120 stage geometry, and current blockers in an immutable atomic
  checkpoint. The former opaque `source_custody_sha256` field is not accepted.
- Reopening identical bytes is resume; changed bytes fail closed.

### S01_ROOT_PROGRAM

S01 is unreachable until the capability interface proves all of:

- producer identity is `fresh_own_lineage_semantic_root_y1_v1`;
- the producer implements the packet-bound canonical source-lineage manifest
  callable; a capability boolean or token list is not lineage evidence;
- the root is not a label-mask/palette realization;
- label/topology is only one factor in a scorer-native RGB realization with
  appearance, chroma gauge, parallax gauge, and an irreducible RGB quotient
  seam;
- exact post-R closure is declared for both SegNet and PoseNet;
- encoder teachers are quarantined and the public receiver is scorer-free;
- the public codec section is a complete exact-hashed runtime tree containing
  a real `inflate.sh`, not a private module inflater;
- the exact canonical G17 whole-object state/receipt closes.

Every compile call receives the sealed lineage context: exact config identity,
compiler source, public runtime tree, and strict G46 audit/primary
receipt/source-video/SegNet/target-label records. The producer returns canonical
manifest bytes binding the emitted packet SHA-256 to that exact context. Every
record has an existing regular-file path, byte count, SHA-256, typed role, and
packaging/dependency classification. The runner recomputes the record closure
and manifest seals, requires the complete core role set, and refuses any
candidate dependency on V15, C1, MS1/MS2R, G85, G57, PR86, or PR130. G46,
source, and scorer artifacts remain encoder-only and outside the archive.

The five stages are chronological immutable cumulative stages over pair ranges
`0:120`, `120:240`, `240:360`, `360:480`, and `480:600`. Each stage must emit:

1. a parse/re-emit-identical `SemanticRootY1V1` packet;
2. a complete public `archive.zip`, not a delta or private intermediate;
3. public archive parse-back identity;
4. two byte-identical full-video decodes by the actual public `inflate.sh`,
   each from a separate clean root with a separately safe-extracted archive and
   copied exact runtime tree;
5. full public-process evidence preserving exact argv, cwd, environment,
   elapsed time, return code, timeout verdict, stdout, stderr, archive-member
   census, output census, and hashes; each public invocation has the contest
   1800-second timeout;
6. an import guard that refuses repository source outside the copied public
   runtime, preventing editable-install or local-package leakage;
7. a batch-16 `upstream/evaluate.py` report over all 600 samples plus a durable
   process receipt preserving stdout/stderr/argv/elapsed/return code on success,
   failure, or timeout;
8. the canonical packet-bound source-lineage manifest and an exact G17
   whole-object authority receipt that carries those same manifest bytes;
9. an immutable atomic stage checkpoint chained to the prior checkpoint.

`--resume-from` is mandatory for S01/status. Resume validates the full archive
public-process receipt, lineage manifest, and G17 receipt bytes before
continuing. The two large raw trees are stage scratch only. After the scorer
report and stage checkpoint are durable, the runner writes an immutable
pre-delete certificate containing every scratch path/byte count/SHA and exact
rebuild inputs, removes the scratch tree, and writes a deletion-completion
receipt. Interrupted partial deletion resumes idempotently by proving every
remaining file is an exact subset of the pre-delete certificate. No proxy,
partial, component-only, historical-replay, private-inflater, or
private-receiver row is admitted. All rows remain explicitly research-only,
noncandidate, and nonscore-claim.

## Selection

After five complete rows, selection is the minimum of the exact coupled contest
score, with archive bytes and SHA-256 only as deterministic ties. The dynamic
target is reloaded and verified at selection time. Independent Seg/Pose/rate
thresholds are forbidden.

## Current executable result

S00 custody is executable when retained G46 artifacts are mounted. A low-level
`SemanticRootY1V1` type may exist independently, but S01 remains correctly
blocked until one module satisfies the complete v2 compiler/lineage/G17
interface and one clean, self-contained public runtime tree with actual
`inflate.sh` is present under the configured exact identities. No run, exact
row, score claim, or pointer movement is produced by this landing.
