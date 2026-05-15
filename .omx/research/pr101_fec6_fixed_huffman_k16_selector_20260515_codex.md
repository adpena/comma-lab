# PR101 FEC6 Fixed-Huffman K16 Selector - 2026-05-15

## Scope

Build a byte-closed PR101 selector candidate that tests whether the larger K16
film-grain/selector palette can beat the current K8 local basin once its
charged selector stream is compressed with a fixed Huffman codebook.

This is not a score claim. It is a candidate archive plus full-frame parity
proof and a dispatched exact `[contest-CPU]` eval.

## Implementation

Commit:

- `cf2a5e2269550406d5381b1abede0b70e28f41ce`
- `https://github.com/adpena/comma-lab/commit/cf2a5e2269550406d5381b1abede0b70e28f41ce`

Files changed:

- `tools/build_pr101_frame_exploit_selector_packet.py`
- `src/tac/tests/test_frame_exploit_selector_packet.py`

Added codec:

- `--compact-selector-codec fec6_fixed_huffman_k16`
- wire magic: `FEC6`
- fixed K16 palette:
  - `none`
  - `frame0_blue_chroma_amp_1`
  - `frame0_blue_chroma_amp_3`
  - `frame0_luma_bias_+1`
  - `frame0_luma_bias_-1`
  - `frame0_luma_bias_-2`
  - `frame0_luma_bias_-4`
  - `frame0_rgb_bias_m2_p1_p1`
  - `frame0_rgb_bias_m4_p2_p2`
  - `frame0_rgb_bias_p0_m1_p1`
  - `frame0_rgb_bias_p0_m2_p2`
  - `frame0_rgb_bias_p0_p1_m1`
  - `frame0_rgb_bias_p0_p2_m2`
  - `frame0_rgb_bias_p2_m1_m1`
  - `frame0_rgb_bias_p4_m2_m2`
  - `frame0_roll_dx+0_dy+1`

Tests:

- `.venv/bin/ruff check tools/build_pr101_frame_exploit_selector_packet.py src/tac/tests/test_frame_exploit_selector_packet.py`
- `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_frame_exploit_selector_packet.py -q`

Result: `20 passed`.

## Candidate Artifact

Clean build artifact:

- archive: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip`
- archive_sha256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- archive_bytes: `178517`
- manifest: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/packet_manifest.json`
- manifest_sha256: `7d3a639487b4313b57978c4514cb68eb8f947d6b77274f4a8d4500925a48ede2`
- runtime_tree_sha256: `12d4315dcbf0943f07fcd357eaf06b126a999c252f8edeb2681179831248df04`
- Modal CPU extractor runtime_tree_sha256: `f67b5b52ca1f11e1a582c53965d88ef738bef86d425b82abdf2e98f3f3fd9166`
- runtime_content_tree_sha256: `6811f28c2116757851b4a6e68a5bdefd7866b4da1867eb13b3c62405de8834df`

Selector payload:

- selector_code_bits_total: `1944`
- selector_index_bytes: `243`
- selector_payload_bytes: `249`
- archive_byte_delta_vs_source: `259`
- selector_payload_sha256: `fc5c431b5d793c33e2f320076fe6f0dd76c2d91e3826ae4b05abfb4f86f453ca`

Proxy only:

- proxy_axis: `PR101 MPS/macOS proxy only`
- selector_score_proxy_charged_formula: `0.19206196565708117`
- selector_score_proxy_uncharged_formula: `0.19188950818822254`
- score_claim: `false`

## Parity Proof

Full-frame same-runtime parity against FEC3 K16:

- source archive: `experiments/results/pr101_frame_exploit_selector_fec3_compact_exact_k16_cpu_overlay_20260515_codex/archive.zip`
- source archive_sha256: `4652e0c19c89d9fa3c55af1a17517d88134f61f6c57a0a7cb23b8bb77ac68989`
- candidate archive_sha256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- proof: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/fec3_k16_fec6_k16_full_streaming_parity.json`
- proof_sha256: `67c3f993f62488ed1351a9b437fe9c7d1828d3321827af5ab19cda97d5488e86`
- full_frame_inflate_output_parity_claim: `true`
- total_frames: `1200`
- total_bytes: `3662409600`
- streaming_raw_sha256: `d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c`

## Exact Eval Dispatch

Exact `[contest-CPU]` eval dispatched:

- lane_id: `lane_pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515`
- Modal call_id: `fc-01KRMNFH1BSMH37VCRCM8RVDPV`
- output_dir: `experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08`
- status at ledger write: `pending`
- recovery command:

```bash
.venv/bin/python tools/recover_modal_auth_eval.py --output-dir /Users/adpena/Projects/pact/experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08
```

Two fail-closed dispatch issues were recorded before the successful detached
spawn:

- `failed_runtime_tree_hash_mismatch`: CPU extractor runtime tree is
  `f67b5b52ca1f11e1a582c53965d88ef738bef86d425b82abdf2e98f3f3fd9166`, while
  the generic uploaded runtime tree in the packet manifest is
  `12d4315dcbf0943f07fcd357eaf06b126a999c252f8edeb2681179831248df04`.
  Content hash matched.
- `failed_refused_dispatch_duplicate_preclaim`: the CPU wrapper owns the claim
  lifecycle and refused a manual pre-claim as an active conflict. The successful
  dispatch let the wrapper claim and spawn.

## Classification

`score_claim=false`

The K8 selector byte-only path is exhausted: FEC5 K8 is already within a few
bytes of the empirical entropy bound and cannot save the 100+ bytes needed to
cross `0.192` with unchanged components.

FEC6 K16 is different: it combines a larger selector palette with a fixed
Huffman codebook. The resulting byte-closed archive is the same size as the
older K8/FEC3 exact-CPU near-miss (`178517` bytes) but decodes to K16-selected
frames. Exact CPU is required to decide whether the K16 component shift survives
outside proxy space.

## 2026-05-15 Exact CPU Recovery

Recovered Modal CPU call `fc-01KRMNFH1BSMH37VCRCM8RVDPV`:

- result review:
  `.omx/research/pr101_fec6_fixed_huffman_k16_cpu_result_review_20260515_codex.json`
- eval JSON:
  `experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/contest_auth_eval.json`
- eval JSON sha256:
  `e82e1b46c61f72e17366e52e64bfa7ccdbbf8058f2363e80d4eb914a9485b6bd`
- archive sha256:
  `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- archive bytes: `178517`
- Modal CPU runtime tree sha256:
  `f67b5b52ca1f11e1a582c53965d88ef738bef86d425b82abdf2e98f3f3fd9166`
- runtime content tree sha256:
  `6811f28c2116757851b4a6e68a5bdefd7866b4da1867eb13b3c62405de8834df`
- samples: `600`
- axis: `[contest-CPU]`
- `avg_segnet_dist`: `0.00056029`
- `avg_posenet_dist`: `0.00002943`
- canonical formula score:
  `0.1920513168811056`

Formula check:

```text
100 * 0.00056029
+ sqrt(10 * 0.00002943)
+ 25 * 178517 / 37545489
= 0.1920513168811056
```

Classification:

- Legitimate score movement on the `[contest-CPU]` axis only.
- `score_claim=false`; `promotion_eligible=false`.
- FEC6 K16 improves the previous FEC3 K8 exact CPU near-miss
  (`0.19209788683213053`) by `-0.0000465699510249`.
- It still misses the operator `<0.192` threshold by
  `0.0000513168811056`, about `77.1` archive bytes at unchanged components.
- CUDA remains unclosed for FEC6 K16 at this point; the FEC3 K8 CUDA run scored
  `0.22626723761043824`, so CPU-positive frame-exploit modes must not be
  inferred to CUDA.

Hardening landed after recovery:

- `tools/build_pr101_frame_exploit_selector_packet.py` now emits a separate
  `modal_cpu_uploaded_runtime_tree_sha256` and uses it for the CPU command
  template.
- Regression coverage in `src/tac/tests/test_frame_exploit_selector_packet.py`
  asserts the CPU template does not reuse the CUDA Modal upload root hash.

## 2026-05-15 Exact CUDA Recovery

Recovered Modal T4 CUDA call `fc-01KRMP4ZM5J8P1H3R2JN96VSC5`:

- result review:
  `.omx/research/pr101_fec6_fixed_huffman_k16_cuda_result_review_20260515_codex.json`
- eval JSON:
  `experiments/results/modal_auth_eval/archive_6bae0201fb08/contest_auth_eval.json`
- eval JSON sha256:
  `e7b64d010ad1b68a07d18304bec32869f156ed1f7da105efd25876a969e4a9b8`
- archive sha256:
  `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- archive bytes: `178517`
- Modal CUDA runtime tree sha256:
  `12d4315dcbf0943f07fcd357eaf06b126a999c252f8edeb2681179831248df04`
- runtime content tree sha256:
  `6811f28c2116757851b4a6e68a5bdefd7866b4da1867eb13b3c62405de8834df`
- samples: `600`
- axis: `[contest-CUDA]`
- `avg_segnet_dist`: `0.00066299`
- `avg_posenet_dist`: `0.00016846`
- canonical formula score:
  `0.22621002169349796`

Formula check:

```text
100 * 0.00066299
+ sqrt(10 * 0.00016846)
+ 25 * 178517 / 37545489
= 0.22621002169349796
```

Classification:

- Legitimate small CUDA improvement, non-promotable.
- FEC6 K16 improves FEC3 K8 exact CUDA (`0.22626723761043824`) by
  `-0.0000572159169403`.
- The paired CPU/CUDA gap is `0.03415870481239236`.
- This confirms the selector/grain family is not a no-op, but CPU-positive
  selector modes transfer weakly to CUDA. Future work must be CUDA-in-loop or
  explicitly paired CPU/CUDA water-fill, not CPU/MPS proxy promotion.
