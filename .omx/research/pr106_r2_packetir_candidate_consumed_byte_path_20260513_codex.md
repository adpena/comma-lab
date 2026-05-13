# PR106/R2 PacketIR Candidate Consumed-Byte Path (2026-05-13)

## Scope

Strengthen the canonical PR106/R2 PacketIR path for latent-sidecar compression
candidates. This landing is local custody/tooling only: no dispatch, no score
claim, no promotion, and no SIREN trainer/provider edits.

## Code Surfaces

- `src/tac/packet_compiler/pr106_sidecar_packet.py`
  - Added `decode_pr106_sidecar_packet_dim_delta(...)`.
  - Added `build_pr106_sidecar_recode_candidate_packet(...)`.
  - Added `emit_pr106_sidecar_recode_candidate_archive(...)`.
  - Added `pr106_sidecar_recode_candidate_manifest(...)`.
  - Extended consumed-byte section rows with explicit `offset_start`,
    `byte_count`, and `offset_end_exclusive` aliases while preserving the
    existing `offset` / `bytes` / `end_offset` contract.
- `tools/profile_pr106_latent_sidecar_recode.py`
  - Candidate rows now carry PacketIR identity and consumed-byte proof status
    when the source is a full PR106 sidecar archive.
  - Added `--emit-runtime-candidates-dir` for deterministic archive plus JSON
    manifest emission for candidates whose runtime decoder already exists.
- `src/tac/tests/test_pr106_latent_sidecar_recode.py`
  - Covers candidate PacketIR consumed-byte proofs and emitted non-promotable
    runtime-candidate archives/manifests.

## Proof Status

Identity proof:

- `tools/prove_pr106_packetir_identity.py` passes on
  `submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip`.
- Expected archive SHA-256:
  `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`.
- `packet_ir_identity_passed=true`.
- `score_claim=false`.
- `ready_for_exact_eval_dispatch=false`.

Candidate proof:

- The PR106/R2 `0x01` source archive can deterministically emit the existing
  PR101 ranked/no-op `0x02` sidecar candidate.
- The emitted candidate member payload matches
  `submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip` member bytes.
- Candidate manifests include:
  - source and candidate packet payload SHA-256s;
  - semantic dim/delta SHA-256s;
  - source and candidate PacketIR consumed-byte proofs;
  - contiguous section offsets and byte counts;
  - per-section SHA-256s;
  - `runtime_consumption_claim=false`;
  - `score_claim=false`;
  - `promotion_eligible=false`;
  - `ready_for_exact_eval_dispatch=false`.

## Claim Boundary

The new candidate path proves lossless sidecar semantic equivalence and
PacketIR parse->reemit identity for supported sidecar formats. It does not
prove full-frame inflate parity for a newly emitted candidate, and it does not
replace exact auth eval.

Required next proof before score or promotion language:

1. Runtime decode/apply proof for the emitted candidate archive and exact
   runtime surface.
2. Full-frame same-runtime parity or same-runtime auth eval where equivalence
   language is needed.
3. Claimed exact `[contest-CUDA]` auth eval with archive/runtime custody,
   dispatch claim closure, and adjudication.

## Solver Wire-In

- `sensitivity-map contribution`: N/A. This landing changes byte-custody and
  candidate manifest plumbing only; no component-distance empirical anchor.
- `pareto constraint`: Non-binding until exact CUDA. Candidate rows expose byte
  deltas and exact blockers, but `ready_for_exact_eval_dispatch=false`.
- `bit-allocator hook`: N/A. The path consumes already selected sidecar
  corrections; it does not alter per-pair importance allocation.
- `cathedral autopilot dispatch hook`: Not registered for dispatch. The emitted
  manifests are explicitly non-promotable and list exact-eval blockers.
- `continual-learning posterior update`: N/A. No empirical anchor was produced.
- `probe-disambiguator`: Existing recode profile keeps multiple candidate
  grammars side-by-side; unsupported grammars remain parser-only until a runtime
  decoder exists.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_pr106_latent_sidecar_recode.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_prove_pr106_packetir_identity_tool.py -q
```

Result: `28 passed`.

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_all_lanes_pr106_sidecar_runtime_gate.py \
  src/tac/tests/test_pr106_latent_sidecar_recode.py -q
```

Result: `21 passed`.

```bash
.venv/bin/ruff check \
  src/tac/packet_compiler/pr106_sidecar_packet.py \
  src/tac/packet_compiler/__init__.py \
  tools/profile_pr106_latent_sidecar_recode.py \
  src/tac/tests/test_pr106_latent_sidecar_recode.py
```

Result: `All checks passed!`.

```bash
.venv/bin/python tools/prove_pr106_packetir_identity.py \
  --archive submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip \
  --expected-archive-sha256 c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383
```

Result: exit `0`.
