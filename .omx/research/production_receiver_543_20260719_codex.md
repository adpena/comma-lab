# Task #543 production receiver/archive landing

Date: 2026-07-19
Lane: `production_receiver_543_20260719`
Verdict: **MEASURED STRUCTURAL BYTE-CLOSE PASS — NON-SCORE, NON-PROMOTABLE**
Authority axis: `[macOS-CPU structural/non-score]`
Frontier pointer: `0.1910828242 [contest-CPU]` **UNMOVED**
Launch state: `launch_ready=false`; no paid dispatch, evaluator, submission, or
pointer mutation was authorized or performed.

## Outcome

The landing adds a scorer-free production archive/receiver for the exact
factor-2 integer lattice and wires factor 2 into the structural V10 compiler.
The archive is one deterministic `ZIP_STORED` member, `0.bin`, containing a
canonical JSON header and ordered big-endian length-prefixed sections. The
required `y_description` and zero-byte `repeat-frame1` policy are implemented;
the optional quotient residual is an ordered sparse signed-int16 record stream.
Raw-y and Brotli-y are executable codecs, while `witness-y-stub` deliberately
remains a typed fail-closed placeholder.

The production inflater performs exact stream consumption, canonical-header and
section validation, byte/hash accounting, ZIP compression refusal, scorer-free
integer realization, exact numerator verification, write-once per-pair stages,
`.partial` size/hash validation, atomic raw promotion, and deterministic tree
hashing. Large-file hashes are streamed. Both builder and inflater perform
storage preflight; the inflater preserves each pair stage for disk resume.

This follows the craft handoff's requirement to ship implementation plus
verification and an honest handoff rather than a chat-only claim:
`docs/operating_manual_craft_handoff.md`.

## MEASURED real-pair receipt

The authoritative machine-readable receipt is
`.omx/research/production_receiver_543_byteclose_receipt_20260719.json`.
Its external byte custody is under
`/Volumes/VertigoDataTier/pact/evidence/production_receiver_543_20260719/`.
The final-code evidence uses `archive_raw_v2`, `decode_c_v2`, and `decode_d_v2`;
the earlier `v1` measurement roots are superseded but retained to avoid signal
loss.

- **MEASURED input:** 12 real completed factor-2 pairs, shape
  `[12,384,512,3]`, SHA-256
  `39c8e255fd27b89f5fab35f71c25a74de2d674b66b5c713684eab3f593f2d48b`.
- **MEASURED archive:** 7,079,287 bytes, SHA-256
  `a151ccd8e0dee2e2229fa5048689499ad3dbf1f5ba041917c4aa9459c8e5e997`;
  packet 7,079,179 bytes, SHA-256
  `6aa02fc6ca8b9688efb0296a429a28cc6c234d788dcc5641ac0369c39d613f20`.
- **MEASURED double decode:** 4.53 s and 4.50 s real wall; both produced
  73,248,192 raw bytes, raw SHA-256
  `9ebe5e4f16434aa0eaafb9f2710fe1755d67e02f4c0bd9def4606b89b1e827b7`,
  and tree SHA-256
  `e452c5b7722c3df7d057df5fbb307d071829de85198af8375f0c0a8aafd8e29e`.
- **MEASURED exactness, independently re-derived from raw bytes:** each decode
  has `14,155,776 / 14,155,776` exact integer values satisfying
  `A_num(frame)=denominator*y` across both frames of all 12 pairs. Frame0 equals
  frame1 at `36,624,096 / 36,624,096` camera values. Byte `cmp` returned zero.
- **MEASURED tests:** the exact three-file acceptance command passes
  `183 passed in 19.40s`; focused lint on the new production surfaces and
  `git diff --check` pass.
- **DERIVED n600 runtime:** slower n12 wall `4.53 s * 50 = 226.5 s = 3.775
  min`, giving `7.947x` headroom against 30 minutes. This is linear projection,
  not a measured full-n600 run.
- **DERIVED n600 bulk:** raw output 3,662,409,600 bytes; preserved stages plus
  final raw preflight 7,325,867,776 bytes. The raw-y payload alone scales to
  353,894,400 bytes, about `9.426x` the 37,545,489-byte reference. Therefore
  raw-y is a structural carrier, not a byte-frontier candidate.

No scorer, scorer weight, GT argmax table, source-frame lookup, Torch import,
MLX, or MPS path exists in receiver decode. The declared
`native-cpu-torch-f32-first-max-class-index.v1` policy identifies the charged
encode-side y-selection contract; this task consumes already-custodied uint8 y
planes and does not recompute logits.

## Triality

### DSL

`Factor2IntegerScorerPlane` is a counted V10 instruction owning factor `2`, with
exact fields `y_uint8`, `camera_shape`, `scorer_shape`, and
`receiver_contract_id`. Its authoritative handler performs the real integer
realization and emits factor-2 custody. `IMPLEMENTED_FACTOR_IDS` now includes
`2`; `MISSING_FACTOR_IDS` remains exactly `("10",)`. All compiler outputs keep
`launch_ready=false`, `score_claim=false`, and `promotion_eligible=false`.

### DAG

```text
real uint8 y + provenance
  -> storage preflight
  -> canonical header + length-prefixed sections
  -> deterministic ZIP_STORED archive + parse-back
  -> exact packet/header/section/hash/trailing-byte checks
  -> scorer-free y decode
  -> factor-2 disjoint-support uint8 realization
  -> exact numerator/nullspace verification
  -> write-once pair stage + state
  -> .partial full raw + streamed size/hash proof
  -> atomic <video>.raw promotion
  -> independent second root + raw/tree equality receipt
```

The executable readiness delta is factor 2 `HAVE`; factor 10 remains `MISSING`.
No downstream launch or score gate is cleared by this structural result.

### Equations

For scorer location `(i,j)`, row/column tap numerators are exact integers with
common denominator

`D = D_row * D_col`.

The canonical feasible point sets every owned camera tap in the disjoint
support to the charged byte `y[i,j,c]`, and every unowned tap to zero. Because
each axis's integer coefficients sum to its denominator,

`A_num(x)[i,j,c] = sum_(r,k) n_row[r] n_col[k] x[r,k,c] = D*y[i,j,c]`.

An optional sparse residual `q` is admitted only after saturating uint8
application and a fresh exact check establishes

`A_num(x + q) = D*y`.

The initial generic frame0 policy is `x0 = x1`; it costs zero video-derived
payload bytes and is explicit in the header.

## Round-1 adversarial self-review

1. **Smuggling:** all charged y and residual bytes are inside the counted
   archive sections. Frame0 repeat is generic and zero-byte. No scorer/source
   data is hidden in code or loaded at decode. PASS for this raw-y instance.
2. **Parser/ZIP:** exact keys, canonical JSON, exact integer types, ordered
   sections, framing lengths, hashes, counted totals, decoded sizes, trailing
   bytes, one-member archive, encryption, and ZIP compression are fail-closed.
   Malformed geometry is bounded before allocation. PASS.
3. **Nondeterminism:** no RNG, unordered serialization, clock, thread pool,
   float decision, or backend kernel affects receiver bytes. Canonical JSON,
   fixed ZIP metadata, integer arithmetic, ordered residuals, and ascending pair
   assembly are deterministic. PASS.
4. **Dictionaries/order:** header keys are canonical-sorted on wire; sections
   and residual records have fixed order; output tree hashing sorts relative
   paths. PASS.
5. **Float/threading:** receiver realization and verdict use uint8/int64 only.
   NumPy vectorization replaces the original per-pixel Python loop but does not
   introduce reduction-order ambiguity in support fill. Exact numerator sums
   are int64. PASS.
6. **Double decode:** independent roots have identical raw and complete tree
   hashes, and independent numerator re-derivation is exact. PASS.
7. **Resumability/disk:** each pair has immutable binary/state legs; a prior
   stage is reopened before progress; final assembly revalidates custody;
   storage preflight and streamed hashing avoid unbounded full-raw allocation.
   PASS.
8. **Scope:** Brotli is synthetic-test-covered but not real-n12 measured here;
   sparse quotient residual is tested but not present in the real-n12 archive;
   raw-y byte competitiveness fails by derivation. These are not promoted into
   broader claims.

## STORES CONSULTED

- Delegated authority:
  `.omx/tmp/codex_runs/production_receiver_543_20260719_20260719T044521Z.wrapped.prompt.txt`
- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`
- `docs/operating_manual_craft_handoff.md`
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`
- `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`
- `.omx/research/receiver_hardening_402_20260711T212832Z.md`
- `.omx/research/v10_compiler_receiver_20260718.md`
- `.omx/research/v10_compiler_receiver_fresh_eyes_20260718.md`
- `.omx/research/v10_uint8_lattice_feasibility_receipt_20260718.json`
- `.omx/research/v10_lattice_rate_verdict_and_composition_20260719.md`
- `.omx/state/lane_registry.json`, `.omx/state/subagent_progress.jsonl`
- Task inbox and broadcast inbox, read before checkpoints and handoff

The sacred source directory
`experiments/results/levelset_n600_witness_20260717T113932Z/` was absent from
this isolated worktree and was not created or mutated.

## Remaining blockers and MAIN handoff

- `raw-uint8-y` is structurally exact but not byte-competitive at n600.
- `witness-y-stub` is not implemented; it refuses decode by design.
- Factor 10 remains missing from the V10 structural compiler.
- No contest-CPU/CUDA exact replay or score was authorized; the pointer remains
  unchanged.

MAIN landing review is required. Review should re-check the canonical receiver
contract import/seal, hostile packet and ZIP bounds, streamed large-file hashes,
factor-2 vectorized support fill, compiler factor-set transition, and the exact
external v2 custody before serializer landing on `main`.
