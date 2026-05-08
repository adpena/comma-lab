# Artifact Disposition Rigor Review - 2026-05-07 Codex

## Scope

Reviewed the completed Worker B/N/S/P artifacts plus the local generated-schema
codec tranche for overclaiming, premature kill/falsification language, and
promotion leakage.

## Disposition Policy Applied

- Exact negative byte artifacts can mark an implementation/config as
  `negative` or `deferred-pending-repair`.
- They do not kill a method family unless there is independent exact evidence
  or a mathematical impossibility proof plus clean adversarial review.
- CPU/proxy/MPS artifacts cannot promote, rank, kill, or anchor score claims.
- Active exact-eval archive anatomy can record bytes/SHA/member structure and
  blockers, but should not parse human logs for score claims when structured
  custody is available.

## Reviewed Artifacts

### Worker S - Shared Static PMF

Artifact: `src/tac/shared_pmf_model.py` and
`tools/pr101_shared_model_pmf_probe.py`.

Verdict: `negative after charged bytes` for this exact static clustered-PMF
artifact only.

Reason:

- exact range payload roundtrip was implemented;
- model overhead was charged;
- result loses to Brotli/Optuna and per-tensor AAC references;
- manifest now uses `artifact_disposition_detail`, not broad
  `falsification`.

This does not falsify learned hyperpriors, Ballé-style predictors, multipass
models, HStack/VStack composition, or model-as-code variants.

### Worker B - CodecOp Bitstream Materializer

Artifact: `src/tac/codec_op_bitstream_materializer.py` and
`tools/materialize_codec_op_bitstream.py`.

Verdict: accepted as a custody/golden-vector bridge.

Reason:

- requires real payload bytes;
- verifies payload SHA/bytes against source metadata;
- emits fail-closed blockers for CPU-only evidence and missing archive
  identity;
- does not treat ADMM planning output as a dispatchable archive.

### Worker N - Packet Compiler Golden Vectors

Artifact: `tools/build_packet_compiler_golden_vectors.py`.

Verdict: accepted as native-port conformance groundwork.

Reason:

- pins deterministic stored-ZIP vectors and duplicate-member negative fixture;
- records ZIP header spans, SHA-256s, charged bytes, and fail-closed duplicate
  behavior;
- does not implement or claim a Rust/Zig/ASM port.

### Worker P - PR103-on-PR106 Active Floor Anatomy

Artifact: `tools/analyze_active_pr103_pr106_floor.py` and
`reports/active_pr103_pr106_floor_anatomy_20260507.json`.

Verdict: accepted as exact archive anatomy/custody surface.

Reason:

- records active archive bytes/SHA and member anatomy;
- records one stored member `0.bin`, payload split, and nested PR103 AC decoder
  section structure;
- suppresses score/distortion values from eval JSON in the report;
- exact-eval blockers are recorded from custody rather than inferred from
  logs.

### Generated HNeRV Schema Codec

Artifact: `src/tac/hnerv_generated_schema_codec.py`.

Verdict: accepted as payload-contract proof only.

Reason:

- generated Stage-D schema now materializes to an `HNGS` blob with deterministic
  decode;
- byte count is recorded but not interpreted as performance;
- runtime loader, local inflate parity, strict packet preflight, lane claim, and
  exact CUDA auth eval remain blockers.

## Open Review Hooks

- Worker H is now scoped to HStack/VStack/multipass hyperprior repair so learned
  hyperprior paths stay active instead of being discarded after the static PMF
  negative artifact.
- Any future use of `FALSIFIED`, `KILL`, `killed`, or method-family retirement
  should be treated as a review-triggering term and backed by independent exact
  evidence or impossibility proof.
