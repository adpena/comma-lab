# Codex Findings - PR101 PacketIR Frontier CPU Authority - 2026-05-19T22:02:44Z

score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
research_only: false

## Operator Question

Does PacketIR / the packet compiler apply to the PR101 frontier and to any
experiment, design, or submission? Have we run it before on our frontier PR
`[contest-CPU]`?

## Finding

Yes, the packet compiler is explicitly intended to be reusable beyond PR106.
`src/tac/packet_compiler/README.md` describes it as reusable byte-grammar and
entropy-coder primitives extracted from public PR101 and PR103 and generalized
across sidecar, per-pair, and per-tensor streams.

Yes, the PR101/FEC6 frontier archive has already been exact-evaluated on
`[contest-CPU]`:

- archive: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip`
- archive SHA-256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- bytes: `178517`
- exact `[contest-CPU]`: `0.1920513168811056`
- paired exact `[contest-CUDA/T4]`: `0.22621002169349796`

The missing surface is narrower: PR101/FEC6 has packet/compiler primitives,
parser manifests, wrapper profiles, and exact eval artifacts, but the artifact
trail I found does not show a prior canonical `compile_packet(...)`,
`build_deterministic_packet.py`, or generalized PacketIR exact-closure run tied
to that `[contest-CPU]` result. The FEC6 artifact was built through the
PR101-specific frame exploit selector builder. It also does not yet have the
same high-level PacketIR candidate authority matrix that PR106 now has. That
matrix is needed so parser identity, runtime byte consumption, compiler
custody, and paired exact-eval authority stay distinct.

## Current PR101/FEC6 Packet Compiler Surfaces

Existing reusable / operator-facing surfaces include:

- `src/tac/packet_compiler/pr101_sidecar_grammar.py`
- `src/tac/packet_compiler/pr101_decoder_byte_maps.py`
- `src/tac/packet_compiler/pr101_fec7_selector.py`
- `tools/build_pr101_runtime_packet.py`
- `tools/build_pr101_frame_exploit_selector_packet.py`
- `tools/build_pr101_frame_conditional_runtime_packet.py`
- `tools/build_a2_sensitivity_weighted_pr101_packet.py`
- `tools/build_fec6_plus_haar_residual_packet.py`
- `tools/pr101_fec6_wrapper_profile.py`
- `tools/profile_pr101_fec6_escape_routes.py`

Current exact result provenance:

- builder: `tools/build_pr101_frame_exploit_selector_packet.py`
- builder manifest: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/packet_manifest.json`
- exact CPU artifact: `experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/contest_auth_eval.json`
- exact CUDA artifact: `experiments/results/modal_auth_eval/archive_6bae0201fb08/modal_cuda_auth_eval_result.json`
- no found deterministic-compiler manifest: no `compile_packet` / `build_deterministic_packet.py` artifact found for this exact archive
- no found generalized PacketIR closure artifact: no PR101/FEC6 `packetir_identity_*`, exact-closure, or deterministic compiler proof artifact found for this exact archive

## Fresh Post-Refactor FEC6 Escape Profile

I reran the PR101/FEC6 post-refactor escape profile against the byte-identical
FEC6 submission archive:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/pr101_fec6_wrapper_profile.py \
  --archive experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip \
  --source-archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip \
  --json-out experiments/results/pr101_fec6_post_refactor_escape_profile_20260519T220201Z/wrapper_profile.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/profile_pr101_fec6_escape_routes.py \
  --fec6-archive experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip \
  --output-dir experiments/results/pr101_fec6_post_refactor_escape_profile_20260519T220201Z
```

Artifacts:

- `experiments/results/pr101_fec6_post_refactor_escape_profile_20260519T220201Z/wrapper_profile.json`
- `experiments/results/pr101_fec6_post_refactor_escape_profile_20260519T220201Z/profile.json`
- `experiments/results/pr101_fec6_post_refactor_escape_profile_20260519T220201Z/profile.md`

Profile SHA-256s:

- `wrapper_profile.json`: `de4ed17b61112686f72a9774372d982e7f679517696ba80401e2c56e0de946ff`
- `profile.json`: `f1bdeab759bb34d635df41fd0f7c319aa03768c6a607ad8d7d0d1190692687ce`
- `profile.md`: `bf93e6f265b8c4b794ffda535178f3b414f8d89584d055f37f11bbab7bf57a76`

## Byte-Only Verdict

Fresh measured byte-only opportunities:

| surface | current bytes | best/floor bytes | realistic saving | verdict |
|---|---:|---:|---:|---|
| PR101 decoder Brotli streams | 162164 | 162162 | 2 | bounded recompress only |
| PR101 latent raw-LZMA | 15387 | 15387 | 0 | filter sweep saturated |
| PR101 latent sidecar | 607 | 603 | 4 | near entropy floor |
| FEC6 selector payload | 249 | 241 | 8 | selector entropy bounded |
| FP11 wrapper | 10 | 0 | 10 | hardcode-only, insufficient |

Conclusion: same-frame realistic byte-saving upper bound is about `16` bytes.
The exact `[contest-CPU]` gap to `<0.192` is `78` bytes. Therefore the next
PR101/FEC6 PacketIR move should not be another opaque byte-only recode. It must
either:

1. change components via CUDA-in-loop / CPU-axis-aware selector waterfill,
2. add a score-affecting residual or procedural packet layer, or
3. broaden the packet compiler to a different substrate / submission family.

## Authority Standard To Add

PR101/FEC6 should get a PR106-style authority matrix with at least these
separate booleans:

- `packetir_identity_proven`
- `deterministic_compiler_identity_proven`
- `runtime_consumes_packetir_bytes`
- `contest_cpu_exact_measured`
- `contest_cuda_exact_measured`
- `paired_exact_same_archive_runtime`
- `ready_for_dispatch`
- `promotion_eligible`

Parser/profile evidence and exact `[contest-CPU]` history already exist.
Promotion authority remains false because current paired CPU/CUDA evidence is
not a score-lowering paired candidate; it is a near-frontier CPU-positive but
CUDA-dominated artifact. Deterministic compiler identity authority should stay
false until the exact FEC6 archive is round-tripped through `compile_packet(...)`
or a generalized PacketIR adapter that delegates to the canonical compiler.

## Recommendation

Treat PacketIR as universal infrastructure for every byte-closed experiment,
design, and submission, but require each family to declare its own authority
matrix before dispatch or promotion. For PR101/FEC6 specifically, the next
engineering artifact should be a `pr101_frontier_packetir_matrix` helper that
records the exact CPU/CUDA anchors above and explicitly refuses to promote
parser/profile-only rows.
