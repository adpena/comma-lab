# PR101/FEC6 Source Payload Anatomy

- Schema: `pr101_fec6_source_payload_anatomy_profile_v1`
- Score claim: `false`
- Score claim valid: `false`
- Promotion eligible: `false`
- Rank/kill eligible: `false`
- Ready for exact eval dispatch: `false`
- Axis tag: `[predicted]`
- Authority contract: `pr101_fec6_source_anatomy_is_read_only_byte_profile_not_score_evidence`
- Archive bytes: `178517`
- Member payload bytes: `178417`
- Source payload bytes: `178158`
- Selector payload bytes: `249`

## Semantic Sections

| section | range | bytes | null bytes | null % | entropy floor | brotli q11 | lzma9e | magic best | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `fp11_magic` | `0-4` | 4 | 4 | 100.00% | 1 | 8 | 60 | 43 | header_not_material_rate_target |
| `source_len_u32le` | `4-8` | 4 | 4 | 100.00% | 1 | 8 | 60 | 41 | header_not_material_rate_target |
| `pr101_decoder_blob` | `8-162172` | 162164 | 39 | 0.02% | 162133 | 162169 | 162232 | 163190 | magic_codec_rewrite_possible_but_current_bytes_are_entropy_saturated |
| `pr101_latent_blob` | `162172-177559` | 15387 | 15387 | 100.00% | 15365 | 15391 | 15448 | 16422 | typed_runtime_adapter_candidate |
| `pr101_sidecar_blob` | `177559-178166` | 607 | 607 | 100.00% | 586 | 611 | 668 | 1642 | typed_runtime_adapter_candidate |
| `selector_len_u16le` | `178166-178168` | 2 | 2 | 100.00% | 1 | 6 | 60 | 29 | header_not_material_rate_target |
| `selector_fec6_payload` | `178168-178417` | 249 | 249 | 100.00% | 220 | 253 | 312 | 1278 | selector_seed_adapter_already_empirically_falsified_on_current_payload |

## Rank-1 Null Run

- Range: `[162171, 178417]`
- Bytes: `16246`
- Components: `pr101_decoder_blob, pr101_latent_blob, pr101_sidecar_blob, selector_len_u16le, selector_fec6_payload`

## Typed Latent Probe

- Current latent blob bytes: `15387`
- Latent raw bytes: `16912`
- Best typed alternative bytes: `15456`
- Best typed alternative delta vs current latent blob: `69`

## Ranked Next Targets

| rank | target | bytes | null % | materialization risk | next action |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `latent_blob_plus_sidecar_semantic_null_span` | 15994 | 100.00% | `high_runtime_adapter_and_exact_eval_required` | Build a runtime-adapter candidate that replaces latent+sidecar encoding, then run byte-mutation/no-op and exact CPU/CUDA eval. |
| 2 | `pr101_sidecar_only_null_span` | 607 | 100.00% | `medium_exact_eval_required` | Probe sidecar-only seed or compact grammar mutations before touching the full latent blob. |
| 3 | `selector_fec6_payload` | 249 | 100.00% | `low_but_already_falsified_by_seeded_selector_probe` | Do not materialize simple selector-seed adapter unless a new predictor beats 249 charged bytes. |
| 4 | `wrapper_headers` | 10 | 100.00% | `header_no_op_and_too_small` | Ignore for score movement; preserve for parser custody. |

## Cascade Relevance

This profile is a negative control for post-hoc magic-codec rewrites of the current PR101/FEC6 archive member bytes. It does not refute magic_codec or magic_codec_dense_streams on newly produced streams; the plausible stack is procedural/DWT/world-model generation first, then dense-stream coding of the residual between generated and empirical bytes.

## Interpretation

The largest null run is not one semantic object: it is the final byte of the PR101 decoder blob, the entire PR101 latent blob, the entire latent sidecar, and the FEC6 selector length/payload. Direct magic-codec/generic rewrites of the current PR101 latent bytes do not beat the incumbent latent blob in this profile (best delta +69 bytes). The next artifact path should therefore be a runtime-adapter/no-op proof for the latent+sidecar semantic null span or a sidecar-only exact-eval probe, not another post-hoc recompression of the already-compressed archive member bytes.
